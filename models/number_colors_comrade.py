import torch
from torch.nn import *
from .library import *


class ComradeModel(Module):
    def __init__(self):
        super().__init__()

        self.layers = Sequential(
            Linear(198, 256),
            ReLU(),
            Linear(256, 256),
            ReLU(),
            Linear(256, 256),
            ReLU(),
            Linear(256, 256),
            ReLU(),
            Linear(256, 192),
        )

    def forward(self, forgery_latent, forgery_categ, orig_categ, num_categ):
        batch_size = forgery_latent.size()[0]

        forgery_categ_onehot = forgery_latent.new_zeros(
            batch_size, num_categ).scatter_(
            1, forgery_categ.unsqueeze(1), 1.0)
        orig_categ_onehot = forgery_latent.new_zeros(
            batch_size, num_categ).scatter_(
            1, orig_categ.unsqueeze(1), 1.0)
        full_vector = torch.cat([
            forgery_latent, forgery_categ_onehot, orig_categ_onehot], dim=1)
        return self.layers(full_vector)


model = ComradeModel()
