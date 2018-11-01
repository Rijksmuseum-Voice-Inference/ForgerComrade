#!/usr/bin/env python3.6

import pdb
import random
import numpy as np
import torch
import util

from describer import Describer
from reconstructor import Reconstructor
from forger import Forger
from comrade import Comrade
from dataset import NumberColorsDataset, display_image


DESCRIBER_FOOTER = "_describer"
RECONSTRUCTOR_FOOTER = "_reconstructor"
FORGER_FOOTER = "_forger"
COMRADE_FOOTER = "_comrade"

NUM_COLORS = 3

example_tensor = torch.tensor(0.0)
if torch.cuda.is_available():
    example_tensor = example_tensor.cuda()


class Parameters:
    def __init__(self):
        self.stage = ""
        self.header = "number_colors"
        self.lr = 0.001
        self.latent_decay = 0.01
        self.batch_size = 32
        self.num_epochs = 0
        self.rand_seed = -1


parser = util.make_parser(Parameters(), "Forger Comrade Model")


def train_analysts(params):
    describer_model = util.load_model(params.header + DESCRIBER_FOOTER)
    describer = Describer(describer_model, NUM_COLORS)
    util.initialize(describer)
    describer.train()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model)
    util.initialize(reconstructor)
    reconstructor.train()

    if example_tensor.is_cuda:
        describer = describer.cuda()
        reconstructor = reconstructor.cuda()

    optim = torch.optim.Adam(
        torch.nn.Sequential(describer, reconstructor).parameters(),
        lr=params.lr)

    data_loader = torch.utils.data.DataLoader(
        NumberColorsDataset(example_tensor),
        batch_size=params.batch_size,
        shuffle=True)

    epoch = 0
    while epoch < params.num_epochs:
        epoch += 1

        loss_sum = 0.0
        loss_count = 0

        print(util.COMMENT_HEADER, end='')

        for orig, orig_categ in data_loader:
            (latent, pred_categ) = describer.describe(orig)
            reconst = reconstructor.reconst(latent)

            categ_loss = describer.categ_loss(pred_categ, orig_categ)
            reconst_loss = reconstructor.reconst_loss(reconst, orig)
            latent_reg = params.latent_decay * \
                (latent ** 2).sum(dim=1).mean(dim=0)

            loss = categ_loss + reconst_loss + latent_reg

            optim.zero_grad()
            loss.backward()
            optim.step()

            print("(" + "|".join([
                "%0.3f" % categ_loss.item(),
                "%0.3f" % reconst_loss.item()]) + ")",
                end=' ', flush=True)
            batch_size = orig.size()[0]
            loss_sum += loss.item() * batch_size
            loss_count += batch_size

        print('')
        loss_mean = loss_sum / loss_count

        metrics = [
            ('epoch', epoch),
            ('loss', round(loss_mean, 3))
        ]
        util.print_metrics(metrics)

    torch.save(
        describer.state_dict(),
        'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth')

    torch.save(
        reconstructor.state_dict(),
        'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth')


def pretrain_manipulators(params):
    describer_model = util.load_model(params.header + DESCRIBER_FOOTER)
    describer = Describer(describer_model, NUM_COLORS)
    describer.load_state_dict(torch.load(
        'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth'))
    describer.eval()

    forger_model = util.load_model(params.header + FORGER_FOOTER)
    forger = Forger(forger_model, NUM_COLORS)
    util.initialize(forger)
    forger.train()

    comrade_model = util.load_model(params.header + COMRADE_FOOTER)
    comrade = Comrade(comrade_model, NUM_COLORS)
    util.initialize(comrade)
    comrade.train()

    if example_tensor.is_cuda:
        describer = describer.cuda()
        forger = forger.cuda()
        comrade = comrade.cuda()

    optim = torch.optim.Adam(
        torch.nn.Sequential(forger, comrade).parameters(),
        lr=params.lr)

    data_loader = torch.utils.data.DataLoader(
        NumberColorsDataset(example_tensor),
        batch_size=params.batch_size,
        shuffle=True)

    epoch = 0
    while epoch < params.num_epochs:
        epoch += 1

        loss_sum = 0.0
        loss_count = 0

        print(util.COMMENT_HEADER, end='')

        for orig, orig_categ in data_loader:
            forgery_categ = torch.randint_like(orig_categ, high=NUM_COLORS)

            forgery = forger.forge(orig, orig_categ, forgery_categ)
            forger_loss = forger.pretrain_loss(forgery, orig)

            orig_latent = describer.latent(orig)
            pretend_latent = comrade.modify_latent(
                orig_latent, forgery_categ, orig_categ)
            comrade_loss = comrade.pretrain_loss(pretend_latent, orig_latent)

            loss = forger_loss + comrade_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            print("(" + "|".join([
                "%0.3f" % forger_loss.item(),
                "%0.3f" % comrade_loss.item()]) + ")",
                end=' ', flush=True)
            batch_size = orig.size()[0]
            loss_sum += loss.item() * batch_size
            loss_count += batch_size

        print('')
        loss_mean = loss_sum / loss_count

        metrics = [
            ('epoch', epoch),
            ('loss', round(loss_mean, 3))
        ]
        util.print_metrics(metrics)

    torch.save(
        forger.state_dict(),
        'snapshots/' + params.header + FORGER_FOOTER + '.pth')

    torch.save(
        comrade.state_dict(),
        'snapshots/' + params.header + COMRADE_FOOTER + '.pth')


def train_manipulators(params):
    describer_model = util.load_model(params.header + DESCRIBER_FOOTER)
    describer = Describer(describer_model, NUM_COLORS)
    describer.load_state_dict(torch.load(
        'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth'))
    describer.eval()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model)
    reconstructor.load_state_dict(torch.load(
        'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth'))
    reconstructor.eval()

    forger_model = util.load_model(params.header + FORGER_FOOTER)
    forger = Forger(forger_model, NUM_COLORS)
    forger.load_state_dict(torch.load(
        'snapshots/' + params.header + FORGER_FOOTER + '.pth'))
    forger.train()

    comrade_model = util.load_model(params.header + COMRADE_FOOTER)
    comrade = Comrade(comrade_model, NUM_COLORS)
    comrade.load_state_dict(torch.load(
        'snapshots/' + params.header + COMRADE_FOOTER + '.pth'))
    comrade.train()

    if example_tensor.is_cuda:
        describer = describer.cuda()
        reconstructor = reconstructor.cuda()
        forger = forger.cuda()
        comrade = comrade.cuda()

    optim = torch.optim.Adam(
        torch.nn.Sequential(forger, comrade).parameters(),
        lr=params.lr)

    data_loader = torch.utils.data.DataLoader(
        NumberColorsDataset(example_tensor),
        batch_size=params.batch_size,
        shuffle=True)

    epoch = 0
    while epoch < params.num_epochs:
        epoch += 1

        loss_sum = 0.0
        loss_count = 0

        print(util.COMMENT_HEADER, end='')

        for orig, orig_categ in data_loader:
            forgery_categ = torch.randint_like(orig_categ, high=NUM_COLORS)

            forgery = forger.forge(orig, orig_categ, forgery_categ)
            (forgery_latent, pred_forgery_categ) = describer.describe(forgery)
            forgery_reconst = reconstructor.reconst(forgery_latent)

            orig_latent = describer.latent(orig)
            pretend_latent = comrade.modify_latent(
                forgery_latent, forgery_categ, orig_categ)
            pretend_reconst = reconstructor.reconst(pretend_latent)

            forgery_categ_loss = describer.categ_loss(
                pred_forgery_categ, forgery_categ)
            forgery_reconst_loss = reconstructor.reconst_loss(
                forgery, forgery_reconst.detach())
            pretend_latent_loss = describer.latent_loss(
                pretend_latent, orig_latent)
            pretend_reconst_loss = reconstructor.reconst_loss(
                pretend_reconst, orig)

            loss = (forgery_categ_loss +
                    forgery_reconst_loss +
                    pretend_latent_loss +
                    pretend_reconst_loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

            print("(" + "|".join([
                "%0.3f" % forgery_categ_loss.item(),
                "%0.3f" % forgery_reconst_loss.item(),
                "%0.3f" % pretend_latent_loss.item(),
                "%0.3f" % pretend_reconst_loss.item()]) + ")",
                end=' ', flush=True)
            batch_size = orig.size()[0]
            loss_sum += loss.item() * batch_size
            loss_count += batch_size

        print('')
        loss_mean = loss_sum / loss_count

        metrics = [
            ('epoch', epoch),
            ('loss', round(loss_mean, 3))
        ]
        util.print_metrics(metrics)

    torch.save(
        forger.state_dict(),
        'snapshots/' + params.header + FORGER_FOOTER + '.pth')

    torch.save(
        comrade.state_dict(),
        'snapshots/' + params.header + COMRADE_FOOTER + '.pth')


def playground(params):
    describer_model = util.load_model(params.header + DESCRIBER_FOOTER)
    describer = Describer(describer_model, NUM_COLORS)
    describer.eval()

    reconstructor_model = util.load_model(params.header + RECONSTRUCTOR_FOOTER)
    reconstructor = Reconstructor(reconstructor_model)
    reconstructor.eval()

    forger_model = util.load_model(params.header + FORGER_FOOTER)
    forger = Forger(forger_model, NUM_COLORS)
    forger.eval()

    comrade_model = util.load_model(params.header + COMRADE_FOOTER)
    comrade = Comrade(comrade_model, NUM_COLORS)
    comrade.eval()

    if example_tensor.is_cuda:
        describer = describer.cuda()
        reconstructor = reconstructor.cuda()
        forger = forger.cuda()
        comrade = comrade.cuda()

    numbers_dataset = NumberColorsDataset(example_tensor)
    numbers_dataset

    def display_reconst(orig):
        orig = orig.unsqueeze(0)
        display_image(reconstructor.reconst(describer.latent(orig))[0])

    def display_forgery(orig, orig_categ, forgery_categ):
        forgery_categ = example_tensor.new_tensor(
            [forgery_categ], dtype=torch.long)
        orig = orig.unsqueeze(0)
        orig_categ = orig_categ.unsqueeze(0)
        forgery = forger.forge(orig, orig_categ, forgery_categ)
        display_image(forgery[0])

    def display_comrade_pretend(orig, orig_categ, forgery_categ):
        forgery_categ = example_tensor.new_tensor(
            [forgery_categ], dtype=torch.long)
        orig = orig.unsqueeze(0)
        orig_categ = orig_categ.unsqueeze(0)
        forgery = forger.forge(orig, orig_categ, forgery_categ)
        pretend_latent = comrade.modify_latent(
            describer.latent(forgery), forgery_categ, orig_categ)
        pretend_reconst = reconstructor.reconst(pretend_latent)
        display_image(pretend_reconst[0])

    def display_comrade_forgery(orig, orig_categ, forgery_categ):
        forgery_categ = example_tensor.new_tensor(
            [forgery_categ], dtype=torch.long)
        orig = orig.unsqueeze(0)
        orig_categ = orig_categ.unsqueeze(0)
        orig_latent = describer.latent(orig)
        forgery_latent = comrade.modify_latent(
            orig_latent, orig_categ, forgery_categ)
        forgery = reconstructor.reconst(forgery_latent)
        display_image(forgery[0])

    try:
        describer.load_state_dict(torch.load(
            'snapshots/' + params.header + DESCRIBER_FOOTER + '.pth'))
        reconstructor.load_state_dict(torch.load(
            'snapshots/' + params.header + RECONSTRUCTOR_FOOTER + '.pth'))
        forger.load_state_dict(torch.load(
            'snapshots/' + params.header + FORGER_FOOTER + '.pth'))
        comrade.load_state_dict(torch.load(
            'snapshots/' + params.header + COMRADE_FOOTER + '.pth'))
    except Exception:
        print("Couldn't load all snapshots!")
        pass

    pdb.set_trace()

    (orig, orig_categ) = numbers_dataset[1573]
    reference = numbers_dataset[573][0]
    forgery_categ = 0
    display_image(orig)
    display_reconst(orig)
    display_forgery(orig, orig_categ, orig_categ)
    display_comrade_pretend(orig, orig_categ, forgery_categ)
    display_image(reference)
    display_reconst(reference)
    display_forgery(orig, orig_categ, forgery_categ)
    display_comrade_forgery(orig, orig_categ, forgery_categ)

    (orig, orig_categ) = numbers_dataset[942]
    reference = numbers_dataset[2942][0]
    forgery_categ = 2
    orig[:, 37:, :] = torch.cat(
        [orig[:, 37:, 75:], orig[:, 37:, :75]], dim=2)
    reference[:, 37:, :] = torch.cat(
        [reference[:, 37:, 75:], reference[:, 37:, :75]], dim=2)
    display_image(orig)
    display_reconst(orig)
    display_forgery(orig, orig_categ, orig_categ)
    display_comrade_pretend(orig, orig_categ, forgery_categ)
    display_image(reference)
    display_reconst(reference)
    display_forgery(orig, orig_categ, forgery_categ)
    display_comrade_forgery(orig, orig_categ, forgery_categ)


def main():
    parsed_args = parser.parse_args()
    params = Parameters()
    util.write_parsed_args(params, parsed_args)

    if params.rand_seed != -1:
        random.seed(params.rand_seed)
        np.random.seed(params.rand_seed)
        torch.random.manual_seed(params.rand_seed)

    if params.stage == "train_analysts":
        train_analysts(params)
    elif params.stage == "pretrain_manipulators":
        pretrain_manipulators(params)
    elif params.stage == "train_manipulators":
        train_manipulators(params)
    elif params.stage == "playground":
        playground(params)
    else:
        print("Unrecognized stage: " + params.stage)
        exit()


if __name__ == '__main__':
    main()
