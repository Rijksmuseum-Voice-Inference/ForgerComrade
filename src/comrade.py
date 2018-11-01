import torch


class Comrade(torch.nn.Module):
    def __init__(self, net, num_categ):
        super().__init__()
        self.net = net
        self.num_categ = num_categ
        self.forward = self.modify_latent
        self.pretrain_loss_fn = torch.nn.MSELoss()

    def modify_latent(self, forgery_latent, forgery_categ, orig_categ):
        return self.net(
            forgery_latent, forgery_categ, orig_categ, self.num_categ)

    def pretrain_loss(self, pretend_latent, orig_latent):
        return self.pretrain_loss_fn(pretend_latent, orig_latent)
