import os
import math
import wandb
import random
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from src.logger import Logger
from src.utils import save_checkpoint, dist_init
from src.optimizer import SGD, LARS
from src.scheduler import StepLinearWarmupCosineAnnealingLR
from src.loader import prepare_imagenet_dataloader as prepare_loader, DINOTransform
from src.loader import prepare_imagenet_val_dataloader as val_loader
from src.builder import build_encoder
from src.utils import statistics, kl_div, convert_to_float
try:
    from src.dali_loader import dali_loader
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False


@torch.no_grad()
def mira(k: torch.Tensor,
         tau: float,
         beta: float,
         iters: int):
    bs = k.size(0) * dist.get_world_size()  # total batch-size

    # fixed point iteration
    k = F.softmax(k / tau / (1 - beta), dim=1)
    temp = k.sum(dim=0)
    dist.all_reduce(temp)
    v = (temp / bs).pow(1 - beta)
    for _ in range(iters):
        temp = k / (v.pow(- beta / (1 - beta)) * k).sum(dim=1, keepdim=True)
        temp = temp.sum(dim=0)
        dist.all_reduce(temp)
        v = (temp / bs).pow(1 - beta)
    temp = v.pow(- beta / (1 - beta)) * k
    target = temp / temp.sum(dim=1, keepdim=True)
    return target


class Model(nn.Module):
    def __init__(
            self,
            out_dim: int = 256,
            proj_hidden_dim: int = 2048,
            num_prototypes: int = 3000,
            arch: str = 'resnet50',
    ):
        super(Model, self).__init__()

        # online model
        self.model = nn.ModuleDict()
        self.model['encoder'] = build_encoder(arch, zero_init_residual=True, cifar=False)
        self.feat_dim = self.model['encoder'].inplanes
        self.model['projector'] = nn.Sequential(nn.Linear(self.feat_dim, proj_hidden_dim, bias=False),
                                                nn.BatchNorm1d(proj_hidden_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
                                                nn.BatchNorm1d(proj_hidden_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(proj_hidden_dim, out_dim))
        self.model['head'] = nn.utils.weight_norm(nn.Linear(out_dim, num_prototypes, bias=False))
        self.model['head'].weight_g.data.fill_(1)
        self.model['head'].weight_g.requires_grad = False

        # ema model
        self.ema_model = nn.ModuleDict({
            'encoder': build_encoder(arch, zero_init_residual=True, cifar=False),
            'projector': nn.Sequential(nn.Linear(self.feat_dim, proj_hidden_dim, bias=False),
                                       nn.BatchNorm1d(proj_hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(proj_hidden_dim,proj_hidden_dim, bias=False),
                                       nn.BatchNorm1d(proj_hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(proj_hidden_dim, out_dim)),
            'head': nn.utils.weight_norm(nn.Linear(out_dim, num_prototypes, bias=False))
        })

        # copy and detach
        for p, mp in zip(self.model.parameters(), self.ema_model.parameters()):
            mp.data.copy_(p.data)
            mp.requires_grad = False

    @torch.no_grad()
    def _update_ema_params(self, m):
        for p, mp in zip(self.model.parameters(), self.ema_model.parameters()):
            mp.data = mp.data * m + p.data * (1. - m)

        # head is shared -- this performs better than using the ema-head
        for p, mp in zip(self.model['head'].parameters(), self.ema_model['head'].parameters()):
            mp.data.copy_(p.data)

    def forward(self, x, momentum=False, m=-1.):
        x = [x] if torch.is_tensor(x) else x

        # bucket
        outs = {}

        # online
        feats = [self.model['encoder'](_x) for _x in x]
        outs['q'] = [self.model['head'](F.normalize(self.model['projector'](_feats), dim=1)) for _feats in feats]

        # EMA
        if momentum:
            with torch.no_grad():
                if m >= 0.:
                    self._update_ema_params(m)
                mfeats = [self.ema_model['encoder'](_x) for _x in x[:2]]
                outs['k'] = [self.ema_model['head'](F.normalize(
                    self.ema_model['projector'](_feats), dim=1)) for _feats in mfeats]
        return outs


def _loss(q, k, target, args):
    # loss
    q = [q] if torch.is_tensor(q) else q
    loss = - (target.repeat(len(q), 1) * F.log_softmax(torch.cat(q,
              dim=0) / args.tau_s, dim=1)).sum(dim=1).mean()

    # outs (including the statistics)
    outs = {'loss': loss}
    ps = F.softmax(q[0] / args.tau_s, dim=1)
    pt = F.softmax(k / args.tau_t, dim=1)
    outs['ent-ps'], outs['ment-ps'], outs['mi-ps'] = statistics(ps)
    outs['ent-pt'], outs['ment-pt'], outs['mi-pt'] = statistics(pt)
    outs['ent-target'], outs['ment-target'], outs['mi-target'] = statistics(
        target)
    outs['kl-pt-target'] = kl_div(pt, target)
    outs['kl-ps-target'] = kl_div(ps, target)
    outs['kl-pt2-target'] = kl_div(F.softmax(k /
                                   args.tau_t / (1 - args.beta), dim=1), target)
    return outs


def loss_func(model_outs, args):
    q, k, targets = model_outs['q'], model_outs['k'], model_outs['targets']
    outs0 = _loss(q=q[1:], k=k[0], target=targets[0], args=args)
    outs1 = _loss(q=q[:1] + q[2:], k=k[1], target=targets[1], args=args)

    # merge
    outs = dict()
    for key in outs0.keys():
        outs[key] = (outs0[key] + outs1[key]) / 2.
    return outs


class Trainer(object):
    def __init__(self, _args):
        # distributed training setting
        if _args.distributed:
            dist_init(local_rank=_args.local_rank)
        self.args = self.configure(_args)

        # set seed
        random.seed(self.args.seed + self.args.local_rank)
        torch.manual_seed(self.args.seed + self.args.local_rank)
        if self.args.benchmark:
            torch.backends.cudnn.deterministic = True

        # build data-loader
        self._build_data_loader()

        # model
        self._build_model()

        # optimizer
        self._build_optimizer()

        # build logger
        # TODO: Resume not available
        self._build_logger()

    @staticmethod
    def get_args_parser():
        parser = argparse.ArgumentParser(conflict_handler='resolve')

        # Benchmark mode
        parser.add_argument('--benchmark', action='store_true', help='set cudnn.deterministic=True')

        # Machine settings
        parser.add_argument('--seed', default=5678, type=int, help='seed')
        parser.add_argument('--data-dir', default=None, help='path to dataset')
        parser.add_argument('--workers', default=64, type=int, help='total number of data loading workers')

        # Boost training
        parser.add_argument('--dali', action='store_true', help='set dali-loader')
        parser.add_argument('--channel-last', action='store_true', help='channel-last-option')

        # Distributed
        parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
        parser.add_argument('--distributed', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])

        # Wandb logging
        parser.add_argument("--name", default='mira', type=str)
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--project", default=None, type=str)

        # Training
        parser.add_argument('--epochs', default=400, type=int, help='number of epochs')
        parser.add_argument('--batch-size', default=4096, type=int, help='batch-size of all GPUs on the current node.')
        parser.add_argument('--freeze-head-epochs', default=1, type=int)

        # Multi-crop augmentation
        parser.add_argument('--num-local-crops', default=6, type=int)

        # EMA
        parser.add_argument('--ema-momentum', default=0.99, type=float)
        parser.add_argument('--ema-momentum-final', default=1., type=float)

        # Optimizer
        parser.add_argument('--optimizer', default='lars', choices=['sgd', 'lars'])
        parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay')

        # Learning rate scheduling
        parser.add_argument('--lr', default=0.3, type=float, help='learning rate')
        parser.add_argument('--min-lr', default=0., type=float, help='final learning rate')
        parser.add_argument('--learning-rate-scaling', default='linear', choices=['linear', 'sqrt'], type=str)      # learning-rate scaling
        parser.add_argument('--warm-up-start-lr', default=0.3, type=float, help='initial warmup learning rate')     # warm-up
        parser.add_argument('--warm-up-epochs', default=10, type=int, help='number of epochs for warm-up')          # warm-up

        # Architecture
        parser.add_argument('--arch', default='resnet50', type=str, help='backbone architecture')
        parser.add_argument('--out-dim', default=256, type=int, help='projection output dimension')
        parser.add_argument('--proj-hidden-dim', default=2048, type=int, help='projection mlp hidden dimension')

        # Hyper-parameters
        parser.add_argument('--num-prototypes', default=3000, type=int)
        parser.add_argument('--tau-s', default=0.1, type=float, help='soft-max temperature')
        parser.add_argument('--tau-t', default=0.225, type=float, help='soft-max temperature')
        parser.add_argument('--beta-initial', default=0.75, type=float)
        parser.add_argument('--beta-final', default=2./3., type=lambda s: convert_to_float(s))
        parser.add_argument('--iters', default=30, type=int)

        return parser

    @classmethod
    def get_args(cls): return cls.get_args_parser().parse_args()

    @staticmethod
    def configure(args):
        args.num_gpus = dist.get_world_size() if args.distributed else 1
        if args.learning_rate_scaling == 'sqrt':
            args.lr *= math.sqrt(args.batch_size) / 4.
        else:
            args.lr *= args.batch_size / 256.
            args.min_lr *= args.batch_size / 256.
        args.batch_size = int(args.batch_size / args.num_gpus)
        args.workers = int((args.workers + args.num_gpus - 1) / args.num_gpus)
        args.name += '-' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        if args.num_local_crops > 0:
            args.dali = False
            print('NVIDIA-DALI not available for num-local-crops > 0')
        return args

    def _build_optimizer(self):
        if self.args.optimizer == 'sgd':
            self.optimizer = SGD(self.trainable_params,
                                 lr=self.args.lr,
                                 weight_decay=self.args.weight_decay,
                                 momentum=0.9)

        elif self.args.optimizer == 'lars':
            self.optimizer = LARS(self.trainable_params,
                                  lr=self.args.lr,
                                  weight_decay=self.args.weight_decay,
                                  momentum=0.9)

        else:
            raise NotImplementedError

        # scheduler
        self.scheduler = StepLinearWarmupCosineAnnealingLR(self.optimizer,
                                                           iters_per_epoch=self.args.iters_per_epoch,
                                                           warm_up_epochs=self.args.warm_up_epochs,
                                                           max_epochs=self.args.epochs,
                                                           min_lr=self.args.min_lr,
                                                           warm_up_start_lr=self.args.warm_up_start_lr)

        # scaler
        self.scaler = GradScaler()

        # step count
        self._step = 0

    def _build_data_loader(self):
        # data-loader
        if self.args.dali and DALI_AVAILABLE:
            self.train_loader = dali_loader(self.args.data_dir,
                                            batch_size=self.args.batch_size,
                                            local_rank=self.args.local_rank,
                                            num_gpus=self.args.num_gpus,
                                            num_workers=min(self.args.workers, 2),
                                            seed=self.args.seed)
            self.train_sampler = None
            self.val_loader = val_loader(self.args.data_dir,
                                         batch_size=self.args.batch_size,
                                         num_workers=self.args.workers,
                                         distributed=self.args.distributed)
        else:
            self.train_loader, self.val_loader, self.train_sampler = prepare_loader(self.args.data_dir,
                                                                                    batch_size=self.args.batch_size,
                                                                                    val_batch_size=self.args.batch_size,
                                                                                    num_workers=self.args.workers,
                                                                                    distributed=self.args.distributed,
                                                                                    transform=DINOTransform(num_local_crops=self.args.num_local_crops))
        self.args.iters_per_epoch = len(self.train_loader)

    def resume(self, resume_path):
        raise NotImplementedError

    def _build_logger(self):
        # wandb logger for local rank 0
        wandb_logger = None
        if self.args.local_rank == 0:
            wandb_logger = wandb.init(entity=self.args.entity,
                                      project=self.args.project,
                                      name=self.args.name,
                                      config=self.args,
                                      reinit=False)

        # logger: for logging from multi-gpu
        self.logger = Logger(wandb_logger=wandb_logger)

    def train_loop(self, epoch):
        # train mode
        self.model.train()

        # configure pbar for local_rank 0
        train_loader = enumerate(self.train_loader)
        if self.args.local_rank == 0:
            train_loader = tqdm(train_loader, total=self.args.iters_per_epoch)
            train_loader.set_description("Train Loop: {}".format(epoch))

        # loop
        for i, batch in train_loader:
            # training step
            outs = self._train_step(batch, epoch)
            self._step += 1

            # logger update
            self.logger.update(outs)

            # step-scheduler
            self.scheduler.step()

    def _end(self):
        if self.logger.wandb_logger is not None:
            self.logger.wandb_logger.finish()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def run(self):
        for epoch in range(self.args.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            # training loop
            self.train_loop(epoch)

            # epoch-wise log
            self.logger.log({'epoch': epoch})

            # epoch-end
            if self.args.local_rank == 0:
                save_checkpoint({
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }, is_best=False, filename='./{}.pth.tar'.format(self.args.name))

        # end loop
        self._end()

    @property
    def beta_momentum(self):
        max_iters = self.args.epochs * self.args.iters_per_epoch
        return self.args.beta_final - 0.5 * (1. + math.cos(math.pi * self._step / max_iters)) * (self.args.beta_final - self.args.beta_initial)

    @property
    def ema_momentum(self):
        max_iters = self.args.epochs * self.args.iters_per_epoch
        return self.args.ema_momentum_final - 0.5 * (1. + math.cos(math.pi * self._step / max_iters)) * (self.args.ema_momentum_final - self.args.ema_momentum)

    def _build_model(self):
        self.model = Model(out_dim=self.args.out_dim,
                           proj_hidden_dim=self.args.proj_hidden_dim,
                           num_prototypes=self.args.num_prototypes,
                           arch=self.args.arch).cuda()

        if self.args.channel_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # wrap DDP for distributed
        if self.args.distributed:
            self.model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(self.model),
                             device_ids=[self.args.local_rank])

    @property
    def trainable_params(self):
        model = self.model.module if self.args.distributed else self.model
        params = [{"name": "encoder", "params": model.model['encoder'].parameters()},
                  {"name": "projector",
                      "params": model.model['projector'].parameters()},
                  {"name": "head", "params": model.model['head'].parameters()},]
        return params

    def _train_step(self, batch, epoch):
        x, targets = batch

        # adjust beta
        self.args.beta = self.beta_momentum

        # device placement
        x = [_x.cuda(non_blocking=True) for _x in x]
        if self.args.channel_last:
            x = [_x.to(memory_format=torch.channels_last) for _x in x]
        targets = targets.cuda(non_blocking=True)
        # mixed precision
        with autocast():
            outs = self.model(x, momentum=True, m=self.ema_momentum)
            outs['targets'] = [mira(k=_k,
                                    tau=self.args.tau_t,
                                    beta=self.args.beta,
                                    iters=self.args.iters) for _k in outs['k']]
            
            # loss
            _loss_outs = loss_func(model_outs=outs, args=self.args)
            loss = _loss_outs['loss']

        # compute gradient and do SGD step
        self.scaler.scale(loss).backward()
        if epoch < self.args.freeze_head_epochs:
            for name, p in self.model.named_parameters():
                if "head" in name:
                    p.grad = None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return {"count": len(targets),
                "Train/loss": loss,
                'Train/ent-ps': _loss_outs['ent-ps'],
                'Train/ment-ps': _loss_outs['ment-ps'],
                'Train/mi-ps': _loss_outs['mi-ps'],
                'Train/ent-pt': _loss_outs['ent-pt'],
                'Train/ment-pt': _loss_outs['ment-pt'],
                'Train/mi-pt': _loss_outs['mi-pt'],
                'Train/ent-target': _loss_outs['ent-target'],
                'Train/ment-target': _loss_outs['ment-target'],
                'Train/mi-target': _loss_outs['mi-target'],
                'Train/kl-pt-target': _loss_outs['kl-pt-target'],
                'Train/kl-pt2-target': _loss_outs['kl-pt2-target'],
                'Train/kl-ps-target': _loss_outs['kl-ps-target']}


if __name__ == '__main__':
    # run
    trainer = Trainer(Trainer.get_args())
    trainer.run()
