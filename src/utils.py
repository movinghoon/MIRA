import torch
import shutil
import torch.distributed as dist

from typing import Sequence, Tuple


# ---------------------------------------------------------------------------------------------------------------------#
# Ref) https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/generic/distributed_util.py
def convert_to_distributed_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"
    if (
            torch.distributed.is_available()
            and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
            and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)


def convert_to_normal_tensor(tensor: torch.Tensor, orig_device: str) -> torch.Tensor:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()
    return tensor


def is_distributed_training_run() -> bool:
    return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and (torch.distributed.get_world_size() > 1)
    )
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Ref) https://github.com/facebookresearch/vissl/blob/main/vissl/utils/distributed_utils.py
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor
# ---------------------------------------------------------------------------------------------------------------------#


# Ref) https://github.com/facebookresearch/dino/blob/main/utils.py
def print_setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# Ref) https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def dist_init(local_rank):
    dist.init_process_group(backend='nccl', init_method="env://", rank=local_rank)
    torch.cuda.set_device(local_rank)
    print_setup_for_distributed(local_rank == 0)
    torch.distributed.barrier()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy_at_k(
        outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def statistics(prob, eps=1e-10):
    prob = concat_all_gather(prob) if dist.is_available() and dist.is_initialized() else prob
    entropy = - (prob * torch.log(prob + eps)).sum(dim=1).mean()
    m_prob = prob.mean(dim=0)   # marginal probability
    m_entropy = - (m_prob * torch.log(m_prob + eps)).sum()   # marginal entropy
    mi = m_entropy - entropy
    return entropy, m_entropy, mi


@torch.no_grad()
def kl_div(p, q, eps=1e-10):
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=1).mean()


def convert_to_float(x):
    try:
        return float(x)
    except:
        num, denom = x.split('/')
        return float(num) / float(denom)
