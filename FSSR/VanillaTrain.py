import time, math
from statistics import mean
from datetime import timedelta

import wandb
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import Logger, construct_name, load_state, clone_state, save_state
from model import EDSR, VGGLoss
from dataset import BasicDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MetaLearner:
    def __init__(self, train_fp, valid_fp, load=None, scale=2, shots=10, bs=1, epochs=20,
        lr=0.0001, meta_lr=0.00001):

        self.model = EDSR(n_resblocks=16, n_feats=64, scale=4).to(device) #r16f64x2
        if load:
            load_state(self.model, load)

        self.optim = optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.loss = Loss.get(loss).to(device)

        train_set = BasicDataset.preset(train_fp, scale, size)
        self.train_dl = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=4)

        self.valid_dls = []
        for valid_fp in valid_fps:
            valid_set = BasicDataset.preset(valid_fp, scale, size)
            valid_dl = DataLoader(valid_set, batch_size=bs, shuffle=True, num_workers=2)
            self.valid_dls.append(valid_dl)

        self.summarize(load, scale, bs, lr, loss, n_resblocks, n_feats)


    def __call__(self, epochs):
        wandb.init(project='tester!', name=self.name, notes=self.repr, config=self.config)
        wandb.watch(self.model)
        self.logs = Logger(self.name + '.logs')

        best_model = clone_state(self.model)
        best_loss = math.inf

        train_losses, valid_losses = [], []
        for epoch in range(epochs):
            print('Epoch %i/%i' % (epoch + 1, epochs))
            train_loss = self.train()
            train_losses.append(train_loss)

            valid_loss = self.validate()
            valid_losses.append(valid_loss)

            eval_loss = mean(valid_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model = clone_state(self.model)

        print('train_loss(%s): %s' \
            % (self.train_set_str, [round(x, 4) for x in train_losses]), file=self.logs)
        for name, losses in zip(self.valid_sets_str, zip(*valid_losses)):
            print('valid_loss(%s): %s' \
                % (name, [round(x, 4) for x in losses]), file=self.logs)
        save_state(best_model, '%s_%i].pth' % (self.name, epochs))


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


    def summarize(self, load, scale, bs, lr, loss, n_resblocks, n_feats):
        self.name = construct_name(name='EDSR-r%if%ix%i' % (scale, n_resblocks, n_feats),
            load=load, dataset=str(train_set), bs=bs, action='vanilla')

        self.config = {
            'dataset': self.train_set_str,
            'model': 'EDSRx%i' % scale,
            'finetuning': load,
            'batch_size': bs,
        }

        self.repr = 'train set: \n   %s \n' % repr(self.train_dl) \
                  + 'valid sets: \n' \
                  +  ''.join(['   %s \n' % repr(s) for s in self.valid_dls]) \
                  + 'finetuning: %s \n' % ('from %s' % load if load else 'False') \
                  + 'scale factor: %s \n' % scale \
                  + 'batch size: %s \n' % bs \
                  + 'learning rate: %s \n' % lr \
                  + 'loss function: %s \n' % loss

    def __repr__(self):
        return self.repr
