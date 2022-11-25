import os
import math
import wandb
import torch
import random
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.distributed as dist

from tqdm import tqdm
from torchvision import transforms

from src.optimizer import SGD
from src.logger import Logger as Logger
from src.utils import accuracy_at_k, dist_init
from src.builder import build_encoder
try:
    from src.dali_loader import dali_classification_loader
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False


# Args
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='path to pth.tar file')
parser.add_argument('--batch-size', default=1024, type=int)
parser.add_argument('--workers', default=4, type=int, help='per gpu # of workers')
parser.add_argument('--distributed', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
parser.add_argument('--data-dir', default=None, type=str)
parser.add_argument('--val-freq', default=1, type=int)
parser.add_argument('--lr', default=0.075, type=float)
parser.add_argument('--weight-decay', default=0., type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument("--arch", default='resnet50', type=str)
parser.add_argument("--entity", default=None, type=str)
parser.add_argument("--project", default=None, type=str)
parser.add_argument('--momentum', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument('--dali', action='store_true', help='set dali-loader')
parser.add_argument('--seed', default=5678, type=int, help='seed')


def load(filename: str, momentum=False):
    ckpt = torch.load(filename, map_location="cpu")
    state_dict = {}

    # load from EMA?
    momentum = momentum and any([('momentum' in key) or ('ema' in key) for key in ckpt['state_dict'].keys()])
    if momentum:
        print("load from momentum")

    for key in ckpt['state_dict'].keys():
        if momentum and (('momentum' not in key) and ('ema' not in key)):
            continue

        elif (not momentum) and (('momentum' in key) or ('ema' in key)):
            continue

        # load encoder only
        if 'encoder' in key:
            state_dict[key.split("encoder.")[-1]] = ckpt['state_dict'][key]
    return state_dict


# epoch-wise cosine scheduling
def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


class LinearClassificationModel(nn.Module):
    def __init__(
            self,
            arch: str = "resnet50",
            num_classes: int = 100,
            cifar: bool = False
    ):
        super().__init__()

        # encoder
        self.encoder = build_encoder(arch, cifar=cifar)
        self.feat_dim = self.encoder.inplanes

        # classifier
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # init
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        return self.classifier(feats)


# train
def train(model, optimizer, loader, epoch, logger):
    loss, acc1, acc5 = [], [], []
    pbar = loader
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            pbar = tqdm(loader, total=len(loader),
                        desc="Epoch:{}".format(epoch))
    else:
        pbar = tqdm(loader, total=len(loader), desc="Epoch:{}".format(epoch))

    for (images, labels) in pbar:
        images = images.cuda(non_blocking=True) if torch.is_tensor(images) else images[0].cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # loss
        logits = model(images)
        _loss = F.cross_entropy(logits, labels)

        # Accuracy
        _acc1, _acc5 = accuracy_at_k(logits, targets=labels, top_k=(1, 5))

        # compute the gradients
        optimizer.zero_grad()
        _loss.backward()

        # step
        optimizer.step()

        # log
        acc1.append(_acc1.item())
        acc5.append(_acc5.item())
        loss.append(_loss.item())
    count, acc1, acc5, loss = len(acc1), np.mean(acc1), np.mean(acc5), np.mean(loss)
    print("Train Loss:{:5.4g}, Acc1:{:5.4g}, Acc5:{:5.4g}".format(loss, acc1, acc5))

    # log
    logger.update({"count": count,
                   "Train/loss": loss,
                   "Train/acc1": acc1,
                   "Train/acc5": acc5})


@torch.no_grad()
def validate(model, loader, epoch, logger):
    loss, acc1, acc5, counts = [], [], [], []
    pbar = loader
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            pbar = tqdm(loader, total=len(loader), desc="Epoch:{}".format(epoch))
    else:
        pbar = tqdm(loader, total=len(loader), desc="Epoch:{}".format(epoch))

    for (images, labels) in pbar:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # loss
        logits = model(images)
        _loss = F.cross_entropy(logits, labels)

        # Accuracy
        _acc1, _acc5 = accuracy_at_k(logits, targets=labels, top_k=(1, 5))

        # log
        num = logits.size(0)
        counts.append(num)
        acc1.append(num * _acc1.item())
        acc5.append(num * _acc5.item())
        loss.append(num * _loss.item())
    counts = sum(counts)
    acc1, acc5, loss = np.sum(acc1) / counts, np.sum(acc5) / counts, np.sum(loss) / counts
    print("Val Loss:{:5.4g}, Acc1:{:5.4g}, Acc5:{:5.4g}".format(
        loss, acc1, acc5))

    # log
    logger.update({"count": counts,
                   "Val/loss": loss,
                   "Val/acc1": acc1,
                   "Val/acc5": acc5})


def main():
    args = parser.parse_args()
    if args.distributed:
        dist_init(args.local_rank)

    # set seed
    random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)

    # distributed configuration
    num_gpus = dist.get_world_size() if args.distributed else 1
    args.lr = args.lr * args.batch_size / 256.
    args.batch_size = int(args.batch_size / num_gpus)

    # data-loader
    if args.dali:
        train_loader = dali_classification_loader(os.path.join(args.data_dir, "train"),
                                                  batch_size=args.batch_size,
                                                  local_rank=args.local_rank,
                                                  num_gpus=dist.get_world_size(),
                                                  num_workers=args.workers,
                                                  seed=args.seed + args.local_rank)
        sampler = None
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_dset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_transform)
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dset) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_dset,
                                                   sampler=sampler,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   drop_last=True,)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_dset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_transform)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dset) if args.distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    # load model
    model = LinearClassificationModel(arch=args.arch, num_classes=len(val_dset.classes))
    model.encoder.load_state_dict(load(filename=args.filename, momentum=args.momentum))
    for param in model.encoder.parameters():
        param.requires_grad = False
    model = model.cuda()
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Optimizer
    _model = model.module if args.distributed else model
    optimizer = SGD(_model.classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # logger
    wandb_logger = None
    if args.local_rank == 0:
        wandb_logger = wandb.init(entity=args.entity,
                                  project=args.project,
                                  name=args.filename.split(
                                      "/")[-1].split(".")[0],
                                  config=args,
                                  reinit=False)
    logger = Logger(wandb_logger=wandb_logger)

    # train
    model.eval()
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, args.lr, epoch=epoch, epochs=args.epochs)

        # train
        train(model, optimizer, train_loader, epoch, logger)

        if (epoch + 1) % args.val_freq == 0 or epoch + 1 == args.epochs:
            validate(model, val_loader, epoch, logger)

            # save
            if args.local_rank == 0:
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                name = './' + args.filename.split("/")[-1].split(".")[0] + '_lincls.pth.tar'
                torch.save(save_dict, name)

            # log
            logger.log({"epoch": epoch})

    # end
    if args.local_rank == 0:
        logger.wandb_logger.finish()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
