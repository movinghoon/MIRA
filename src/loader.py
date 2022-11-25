import os
import random
import torchvision.transforms as transforms

from PIL import ImageFilter, ImageOps, Image
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# Augmentation utils
# ---------------------------------------------------------------------------------------------------------------------#
# Ref) https://github.com/facebookresearch/moco-v3/blob/main/moco/loader.py
class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
# ---------------------------------------------------------------------------------------------------------------------#


# Transform base
# ---------------------------------------------------------------------------------------------------------------------#
class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class MultiCropTransform:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        return list(map(lambda trans: trans(x), self.trans))
# ---------------------------------------------------------------------------------------------------------------------#


# ImageNet transform
# ---------------------------------------------------------------------------------------------------------------------#
class DINOTransform(MultiCropTransform):
    def __init__(self, num_local_crops=0, sizes=(224, 96), scale=0.14):
        # global aug1
        global_aug1 = transforms.Compose([
            transforms.RandomResizedCrop(sizes[0], scale=(scale, 1.), interpolation=Image.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # global aug2
        global_aug2 = transforms.Compose([
            transforms.RandomResizedCrop(sizes[0], scale=(scale, 1.), interpolation=Image.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # local aug
        local_aug = transforms.Compose([
            transforms.RandomResizedCrop(sizes[1], scale=(0.05, scale), interpolation=Image.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        trans = [global_aug1, global_aug2] + [local_aug] * num_local_crops
        super(DINOTransform, self).__init__(trans=trans)


class ImageNetValTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),  # resize shorter
            transforms.CenterCrop(224),  # take center crop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x)
# ---------------------------------------------------------------------------------------------------------------------#


# Dataloader
# ---------------------------------------------------------------------------------------------------------------------#
def prepare_imagenet_dataloader(data_dir: str,
                                batch_size: int,
                                val_batch_size: int,
                                num_workers: int,
                                distributed: bool = True,
                                transform=None):
    transform = DINOTransform() if transform is None else transform

    # train-data-loader
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=not distributed,
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)

    # val-data-loader
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=ImageNetValTransform())
    val_sampler = DistributedSampler(val_dataset) if distributed else None
    val_loader = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            sampler=val_sampler,
                            drop_last=False)
    return train_loader, val_loader, train_sampler


def prepare_imagenet_val_dataloader(data_dir: str,
                                    batch_size: int,
                                    num_workers: int,
                                    distributed: bool = True):
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=ImageNetValTransform())
    val_sampler = DistributedSampler(val_dataset) if distributed else None
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            sampler=val_sampler,
                            drop_last=False)
    return val_loader
# ---------------------------------------------------------------------------------------------------------------------#


# CIFAR related
# ---------------------------------------------------------------------------------------------------------------------#
class CIFARTransform(TwoCropsTransform):
    def __init__(self):
        # aug 1
        aug1 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])

        # aug 2
        aug2 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])

        super(CIFARTransform, self).__init__(aug1, aug2)


class CIFARValTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
        ])

    def __call__(self, x):
        return self.transform(x)


def prepare_cifar_dataloader(data_dir: str,
                             dataset: str,
                             batch_size: int,
                             num_workers: int):
    # dataset from torchvision
    DsetClass = CIFAR10 if dataset.lower() == 'cifar10' else CIFAR100

    # train-data-loader
    train_dataset = DsetClass(data_dir,
                              train=True,
                              download=True,
                              transform=CIFARTransform())
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    # val-data-loader
    val_dataset = DsetClass(data_dir,
                            train=False,
                            download=True,
                            transform=CIFARValTransform())
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)
    return train_loader, val_loader
# ---------------------------------------------------------------------------------------------------------------------#

