import torch
from torch.nn import *
from .library import *


class PartialAvgPool(Module):
    def __init__(self, num_average):
        super().__init__()
        self.num_average = num_average
        self.avg_pool = GlobalAvgPool()

    def forward(self, features):
        batch_size = features.size()[0]
        to_average = features[:, -self.num_average:, :, :]
        return torch.cat([
            features[:, :-self.num_average, :, :].reshape([batch_size, -1]),
            self.avg_pool(to_average)], dim=1)


model = Sequential(
    Dropout(),
    Conv2d(3, 16, 3, stride=2),
    Dropout(),
    ReLU(),
    Conv2d(16, 32, 3, stride=2),
    Dropout(),
    ReLU(),
    Conv2d(32, 64, 3, stride=2),
    Dropout(),
    ReLU(),
    Conv2d(64, 64, 3, stride=2),
    Dropout(),
    ReLU(),
    Conv2d(64, 35, 3, stride=2),
    PartialAvgPool(3)
)
