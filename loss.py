from torch import nn

class WeightedL1Loss(nn.Module):
    def __init__(self, bin_weights):
        super(WeightedL1Loss, self).__init__()
        self.bin_weights = bin_weights

    def forward(self, input, target):
        bins = [t // 5 for t in target]
        weights = [self.bin_weights[b] for b in bins]
        losses = nn.L1Loss(reduction='none')(input, target)
        return torch.dot(weights, losses)
