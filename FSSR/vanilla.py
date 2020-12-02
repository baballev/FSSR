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
        size=(256, 512), loss='L1'):
        self.epochs = epochs
        self.scale = scale
        self.load = load
        self.bs = bs
        self.lr = lr
        self.loss_name = loss
        self.model = EDSR(scale=scale).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)

        if load:
            load_state(self.model, load)

        if loss == 'VGG':
            self.loss = VGGPerceptualLoss().to(device)
        elif loss == 'L2':
            self.loss = F.mse_loss
        elif loss == 'L1'
            self.loss = F.l1_loss
        else:
            raise NotImplementedError('loss function %s not found' % loss)

        train_set = BasicDataset.preset(train_fp, scale, size)
        self.train_dl = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=4)
        self.train_set_str = str(train_set)
        self.train_set_repr = repr(train_set)

        self.valid_dls, self.valids_sets_str, self.valids_sets_repr = [], [], []
        for valid_fp in valid_fps:
            valid_set = BasicDataset.preset(valid_fp, scale, size)
            valid_dl = DataLoader(valid_set, batch_size=bs, shuffle=True, num_workers=2)
            self.valid_dls.append(valid_dl)
            self.valid_sets_str.append(str(valid_set))
            self.valid_sets_repr.append(repr(valid_set))

        self.name = construct_name('EDSRx%i' % scale, load, self.train_set_str, epochs, bs, 'vanilla')
        print(repr(self))

    def __call__(self):
        wandb.init(project='tester!', name=self.name, notes=str(self), config={
            'train_set': self.train_set_str,
            'model': 'EDSRx%i' % scale,
            'finetuning': load,
            'batch_size': bs,
        })

        wandb.watch(self.model)
        self.logs = Logger(self.name + '.logs')

        since = time.time()
        best_model = clone_state(self.model)
        best_loss = math.inf

        train_losses, valid_losses = [], []
        for epoch in range(self.epochs):
            print('Epoch %i/%i' % (epoch + 1, self.epochs))
            train_loss = self.train()
            train_losses.append(train_loss)

            valid_loss = self.validate()
            valid_losses.append(valid_loss)

            eval_loss = mean(valid_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model = clone_state(self.model)

        since = time.time() - since
        print('Summary of training: finished in %s' \
            % timedelta(seconds=int(since)), file=self.logs)
        print('train_loss(%s): %s' \
            % (self.train_set_str, [round(x, 4) for x in train_losses]), file=self.logs)
        for name, losses in zip(self.valid_sets_str, zip(*valid_losses)):
            print('valid_loss(%s): %s' \
                % (name, [round(x, 4) for x in losses]), file=self.logs)

        save_state(best_model, self.name + '.pth')

    def train(self):
        losses = []
        for data in (t := tqdm(self.train_dl)):
            x, y = [d.to(device) for d in data]
            self.optim.zero_grad()
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
            wandb.log({'train_loss_%s' % self.train_set_str: loss})
            t.set_description('Train loss: %.4f (~%.4f)' % (loss, mean(losses)))
        return mean(losses)

    @torch.no_grad()
    def validate(self):
        valid_losses = []
        for valid_dl, name in zip(self.valid_dls, self.valid_sets_str):
            losses = []
            for data in valid_dl:
                x, y = [d.to(device) for d in data]
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                losses.append(loss.item())

            valid_loss = mean(losses)
            valid_losses.append(valid_loss)
            wandb.log({'valid_loss_%s' % name: valid_loss})
            print('valid_loss(%s): %.4f' % (name, valid_loss))
        return valid_losses

    def __repr__(self):
        string = 'train set: \n   %s \n' % self.train_set_repr \
               + 'valid sets: \n   ' \
               +  ''.join(['%s \n' % s[2] for s in self.valid_sets_repr]) \
               + 'finetuning: %s \n' % ('from %s' % self.load if self.load else 'False') \
               + 'scale factor: %s \n' % self.scale \
               + 'batch size: %s \n' % self.bs \
               + 'number of epochs: %s \n' % self.epochs \
               + 'learning rate: %s \n' % self.lr \
               + 'loss function: %s \n' % self.loss_name
        return string
