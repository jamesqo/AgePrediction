import logging

import torch
from torch import nn

from .fds import FDS

print = logging.info

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class SFCN(nn.Module):
    def __init__(self, in_channels, num_classes=1, fds=None):
        super().__init__()

        # Convolutional block sizes
        cfg = [32, 64, 128, 256, 256, 64]

        self.conv1 = ConvBlock(in_channels, cfg[0])
        self.conv2 = ConvBlock(cfg[0], cfg[1])
        self.conv3 = ConvBlock(cfg[1], cfg[2])
        self.conv4 = ConvBlock(cfg[2], cfg[3])
        self.conv5 = ConvBlock(cfg[3], cfg[4])
        self.conv6 = ConvBlock(cfg[4], cfg[5], kernel_size=1)

        self.maxp = nn.MaxPool3d(2)
        self.avg = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Linear(cfg[-1], num_classes)

        self.uses_fds = fds
        if fds:
            self.fds = FDS(
                feature_dim=cfg[-1],
                bucket_num=fds['bucket_num'],
                bucket_start=fds['bucket_start'],
                start_update=fds['start_update'],
                start_smooth=fds['start_smooth'],
                kernel=fds['kernel'],
                ks=fds['ks'],
                sigma=fds['sigma'],
                momentum=fds['momentum']
            )
            self.start_smooth = fds['start_smooth']

    def forward(self, x, targets=None, epoch=None):
        x = self.conv1(x)
        x = self.maxp(x)

        x = self.conv2(x)
        x = self.maxp(x)

        x = self.conv3(x)
        x = self.maxp(x)

        x = self.conv4(x)
        x = self.maxp(x)

        x = self.conv5(x)
        x = self.maxp(x)

        x = self.conv6(x)

        encoding = torch.flatten(self.avg(x), 1)

        encoding_s = encoding
        if self.training and self.uses_fds:
            if epoch >= self.start_smooth:
                encoding_s = self.fds.smooth(encoding_s, targets, epoch)
        x = self.classifier(encoding_s)

        if self.training and self.uses_fds:
            return x, encoding
        else:
            return x
