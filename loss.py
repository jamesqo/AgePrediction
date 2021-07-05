import torch
from torch import nn

class WeightedL1Loss(nn.Module):
    def __init__(self, bin_weights):
        super(WeightedL1Loss, self).__init__()
        self.bin_weights = bin_weights

    def forward(self, input, target):
        sample_bins = [age // 5 for age in target]
        sample_weights = [self.bin_weights[bin] for bin in sample_bins]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float).cuda()
        losses = nn.L1Loss(reduction='none')(input, target).float()
        return torch.dot(sample_weights, losses) / torch.sum(sample_weights)
