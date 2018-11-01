import torch


class Forger(torch.nn.Module):
    def __init__(self, net, num_categ):
        super().__init__()
        self.net = net
        self.num_categ = num_categ
        self.forward = self.forge
        self.pretrain_loss_fn = torch.nn.MSELoss()

    def forge(self, orig, orig_categ, forgery_categ):
        return self.net(orig, orig_categ, forgery_categ, self.num_categ)

    def pretrain_loss(self, forgery, orig):
        return self.pretrain_loss_fn(forgery, orig)
