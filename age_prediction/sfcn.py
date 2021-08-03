import torch
from torch import nn

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
    def __init__(self, in_channels, num_classes=1):
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

    def forward(self, x):
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

        x = torch.flatten(self.avg(x), 1)
        x = self.classifier(x)

        return x
