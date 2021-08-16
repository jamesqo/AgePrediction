import torch
from torch import nn

class SlidingWindow(nn.Module):
    def __init__(self, model, in_channels, window_size):
        super().__init__()

        self.model = model
        self.in_channels = in_channels
        self.window_size = window_size

    def forward(self, x):
        assert x.shape[1] == self.in_channels

        num_windows = self.in_channels - self.window_size + 1
        preds = []
        for start in range(num_windows):
            end = start + self.window_size
            pred = self.model(x[:, start:end, :, :])
            preds.append(pred)
        return torch.mean(preds)
