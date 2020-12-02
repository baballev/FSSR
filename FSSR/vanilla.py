import os,  time,  warnings, math
from statistics import mean
from datetime import timedelta

import wandb
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import Logger, construct_name, load_state, clone_state, save_state
from models import EDSR
from meta import Meta
from datasets import BasicDataset
from loss_functions import VGGPerceptualLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VanillaTrain:
    def __init__(self, train_fp, valid_fps, load=None, scale=8, bs=16, epochs=20, lr=0.0001,
        resize=(256, 512), loss='L1'):
        self.epochs = epochs
        self.model = EDSR(scale=scale)#.to(device)

        if load:
            load_state(self.model, load)

        if loss == 'vgg':
            self.loss = VGGPerceptualLoss().to(device)
        elif loss == 'L2':
            self.loss = F.mse_loss
        else:
            loss = 'L1'
            self.loss = F.l1_loss

        self.optim = optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)

        train_set = BasicDataset(train_fp, scale, augment='augmentor', style=False, resize=resize)
        self.train_dl_name = train_set.name
        self.train_dl = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=4)

        self.valid_dls, self.valid_dl_names = [], []
        for fp in valid_fps:
            valid_set = BasicDataset(fp, scale, augment=False, style=False, resize=resize)
            valid_dl = DataLoader(valid_set, batch_size=bs, shuffle=True, num_workers=2)
            self.valid_dls.append(valid_dl)
            self.valid_dl_names.append(valid_set.name)

        # logging
        self.name = construct_name('EDSRx%i' % scale, load, self.train_dl_name, epochs, bs, 'vanilla-train')
        self.logs = Logger(name + '.log')
        self.repr = 'train set        -> %s \n' % self.train_dl_name \
                  + 'valid sets       -> %s \n' % self.valid_dl_names \
                  + 'finetuning       -> %s %s \n' % ('from %s' % x if x else 'False') \
                  + 'scale factor     -> %s \n' % scale \
                  + 'batch size       -> %s \n' % bs \
                  + 'number of epochs -> %s \n' % self.epochs \
                  + 'learning rate    -> %s \n' % lr \
                  + 'loss function    -> %s \n' % loss

        wandb.init(project='tester!', name=self.name, notes=self.str, config={
            'train_set': self.train_dl_name,
            'model': 'EDSRx%i' % scale,
            'finetuning': load,
            'batch_size': bs,
        })

    def __call__(self):
        wandb.watch(self.model)
        since = time.time()
        best_model = clone_state(model)
        best_loss = math.inf

        train_losses, valid_losses = [], []
        for epoch in range(self.epochs):
            print('Epoch %i/%i' % (epoch + 1, self.epochs))
            avg_loss = self.train()
            train_losses.append(avg_loss)

            valid_loss = self.validate()
            valid_losses.append(avg_loss)

            eval_loss = mean(valid_loss)
            if eval_loss < best_loss:
                best_loss = valid_loss
                best_model = clone_state(self.model)

        since = time.time() - since
        print('Training finished in %s' % timedelta(seconds=int(time_elapsed)))
        print('train_loss_%s: %s' % (self.train_dl_name, train_losses), file=self.logs)
        for dl_name, dl_losses in zip(self.valid_dl_names, zip(*valid_losses)):
            print('train_loss_%s: %s' % (dl_name, dl_losses))

        return best_model


    def train(self):
        losses = []
        for data in (t := tqdm(self.train_dl)):
            x, y = [d.to(device) for d in data]
            self.optim.zero_grad()
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
            self.loss.backward()
            self.optim.step()

            losses.append(loss.item())
            wandb.log({'train_loss_%s' % self.train_dl_name: loss})
            t.set_description('Train loss: %.4f (~%.4f)' % (loss, mean(losses)))
        print('train_loss_%s: %.4f' % (self.train_dl_name, mean(losses)))
        return mean(losses)


    @torch.no_grad()
    def validate(self):
        for valid_dl, dl_name in zip(self.valid_dls, self.valid_dl_names):
            losses = []
            for data in valid_dl:
                x, y = data[0].to(device), data[1].to(device)
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                losses.append(loss.item())
            valid_loss = mean(losses)
            wandb.log({'valid_loss_%s' % dl_name: mean(losses)})
            print('valid_loss_%s: %.4f' % (dl_name, mean(losses)))


    def __str__(self):
        return self.repr
