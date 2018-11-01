from torch.nn import *
import torch.nn.functional as F


class Reshape(Module):
    def __init__(self, *target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, features):
        return features.reshape([-1, *self.target_size])


class Resize(Module):
    def __init__(self, *target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, features):
        return F.interpolate(
            features, self.target_size)


class GlobalAvgPool(Module):
    def forward(self, features):
        (batch_size, channels, _, _) = features.size()
        return features.reshape(batch_size, channels, -1).mean(dim=2)


class PrintSize(Module):
    def forward(self, features):
        print(features.size())
        return features
