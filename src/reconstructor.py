import torch


class Reconstructor(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.reconst_loss_fn = torch.nn.MSELoss()
        self.forward = self.reconst

    def reconst(self, latent):
        return self.net(latent)

    def reconst_loss(self, reconst, original):
        return self.reconst_loss_fn(reconst, original)
