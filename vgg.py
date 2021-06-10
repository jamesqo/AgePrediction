import torch
from torch import nn

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
    def __init__(self, in_channels, num_classes=1):
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

    def forward(self, x):
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

        x = torch.flatten(self.avg(x), 1)
        x = self.classifier(x)

        return x
