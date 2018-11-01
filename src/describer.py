import torch


class Describer(torch.nn.Module):
    def __init__(self, net, num_categ):
        super().__init__()
        self.net = net
        self.num_categ = num_categ
        self.categ_loss_fn = torch.nn.CrossEntropyLoss()
        self.latent_loss_fn = torch.nn.MSELoss()
        self.forward = self.describe

    def describe(self, values):
        descr = self.net(values)
        descr_size = descr.size()[1]
        (latent, categ) = torch.split(
            descr, [descr_size - self.num_categ, self.num_categ], dim=1)
        return (latent, categ)

    def latent(self, values):
        (latent, _) = self.forward(values)
        return latent

    def categ(self, values):
        (_, categ) = self.forward(values)
        return categ

    def categ_loss(self, predict_categ, true_categ):
        return self.categ_loss_fn(predict_categ, true_categ)

    def latent_loss(self, pretend_latent, orig_latent):
        return self.latent_loss_fn(pretend_latent, orig_latent)
