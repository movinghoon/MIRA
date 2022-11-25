import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Optional


# Ref) https://github.com/pytorch/pytorch/blob/fc37e5b3ed0e51a0dff39da6e3a1a3ce16fe2756/torch/optim/_functional.py
def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool = False):
    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0 and param.ndim > 1:    # apply weight-decay except BN params, Bias
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)


# Ref) https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
class SGD(torch.optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            ratio = 1. if 'ratio' not in group else group['ratio']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad.mul(ratio))

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


class LARS(Optimizer):
    """
        https://github.com/facebookresearch/moco-v3/blob/main/moco/optimizer.py
    """
    def __init__(
            self,
            params,
            lr: float = 0.1,
            momentum: float = 0.9,
            weight_decay: float = 0,
            eta: float = 0.001,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eta": eta,
            "exclude": False,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            ratio = 1. if 'ratio' not in g else g['ratio']
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue
                dp = dp.mul(ratio)  # if not None

                if p.ndim > 1 and not g["exclude"]:
                    dp = dp.add(p, alpha=g["weight_decay"])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])