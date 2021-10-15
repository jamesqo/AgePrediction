import torch
from torch import nn

from .fds import FDS

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class VGG8(nn.Module):
    def __init__(self, in_channels, num_classes=1, fds=None):
        super().__init__()

        # Convolutional block sizes
        cfg = [64, 128, 256, 512]

        self.conv11 = ConvBlock(in_channels, cfg[0])
        self.conv12 = ConvBlock(cfg[0], cfg[0])

        self.conv21 = ConvBlock(cfg[0], cfg[1])
        self.conv22 = ConvBlock(cfg[1], cfg[1])

        self.conv31 = ConvBlock(cfg[1], cfg[2])
        self.conv32 = ConvBlock(cfg[2], cfg[2])

        self.conv41 = ConvBlock(cfg[2], cfg[3])
        self.conv42 = ConvBlock(cfg[3], cfg[3])

        self.maxp = nn.MaxPool2d(2)
        self.avg = nn.AdaptiveAvgPool2d(1)

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
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.maxp(x)

        encoding = torch.flatten(self.avg(x), 1)

        encoding_s = encoding
        if self.training and self.uses_fds:
            if epoch >= self.start_smooth:
                encoding_s = self.fds.smooth(encoding_s, targets, epoch)
        x = self.classifier(encoding_s)

        if self.training and self.uses_fds:
            return x, encoding, encoding_s
        else:
            return x, encoding
