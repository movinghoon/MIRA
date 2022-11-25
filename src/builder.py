import torchvision
import torch.nn as nn


def build_encoder(arch: str = "resnet18",
                  zero_init_residual: bool = True,
                  cifar: bool = False):
    encoder = vars(torchvision.models)[arch](zero_init_residual=zero_init_residual)
    encoder.fc = nn.Identity()
    if cifar:
        encoder.conv1 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(2, 2),
                                  bias=False)
        encoder.maxpool = nn.Identity()
    return encoder
