import math
from statistics import mean

import wandb
import torch
from tqdm import tqdm

from .Run import Run
from utils import clone_state, clone

class Train(Run):
    def __init__(self, *args, **kwargs):
        super(Train, self).__init__(*args, **kwargs)
        self.scheduler = None


    def __call__(self, **kwargs):
        super().__call__(**kwargs)

        best = clone_state(self.model), math.inf, -1
        train_losses, valid_losses = [], []
        for epoch in range(self.epochs):
            self.log('Epoch %i/%i' % (epoch + 1, self.epochs))
            train_loss = self.train()
            train_losses.append(train_loss)

            valid_loss = self.validate(step=len(self.train_dl)*(epoch+1))
            valid_losses.append(valid_loss)

            eval_loss = mean(valid_loss)
            if eval_loss <= best[1]:
                if self.wandb:
                    wandb.run.summary['best'] = eval_loss
                best = clone_state(self.model), eval_loss, epoch + 1


        super().terminate(*best)


    def train(self):
        losses = []
        for data in (t := tqdm(self.train_dl)):
            loss = self.train_batch(data)
            losses.append(loss)
            self.step_lr()

            max_grad = max(p.grad.detach().abs().max() for p in self.model.parameters() if p.grad is not None)
            self.log({'train_loss_%s' % self.train_dl: loss, 
                      'lr': self.get_lr(),
                      'max_grad': max_grad})
            t.set_description('Train loss: %.4f (~%.4f)' % (loss, mean(losses)))
        return mean(losses)


    def validate(self, step):
        model = clone(self.model)
        valid_loss = []
        for valid_dl in self.valid_dls:
            losses = []
            for data in valid_dl:
                loss = self.validate_batch(model, data)
                losses.append(loss)
            loss_avg = mean(losses)
            valid_loss.append(loss_avg)
            self.log({f'valid_loss_{valid_dl}': loss_avg}, step=step)
            #print('valid_loss(%s): %.4f' % (valid_dl, loss_avg))
        return valid_loss


    def step_lr(self):
        if self.scheduler:
            self.scheduler.step()


    def get_lr(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        else:
            return self.lr
