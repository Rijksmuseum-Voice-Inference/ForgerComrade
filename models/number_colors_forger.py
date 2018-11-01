import torch
from torch.nn import *
from .library import *


class ForgerModel(Module):
    def __init__(self):
        super().__init__()

        self.layers = Sequential(
            Conv2d(9, 64, 5, stride=2),
            ReLU(),
            Conv2d(64, 128, 5, stride=2),
            ReLU(),
            Conv2d(128, 256, 5, padding=2),
            ReLU(),
            Conv2d(256, 256, 5, padding=2),
            ReLU(),
            Conv2d(256, 128, 5, padding=2),
            ReLU(),
            ConvTranspose2d(128, 64, 5, stride=2, output_padding=(1, 0)),
            ReLU(),
            ConvTranspose2d(64, 3, 5, stride=2)
        )

    def forward(self, orig, orig_categ, forgery_categ, num_categ):
        batch_size, _, height, width = orig.size()

        orig_categ_onehot = orig.new_zeros(batch_size, num_categ).scatter_(
            1, orig_categ.unsqueeze(1), 1.0).reshape(batch_size, 3, 1, 1)
        forgery_categ_onehot = orig.new_zeros(batch_size, num_categ).scatter_(
            1, forgery_categ.unsqueeze(1), 1.0).reshape(batch_size, 3, 1, 1)

        layer_input = torch.cat([
            orig,
            orig_categ_onehot.expand(-1, 3, height, width),
            forgery_categ_onehot.expand(-1, 3, height, width)
        ], dim=1)

        return self.layers(layer_input)


model = ForgerModel()
