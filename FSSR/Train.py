import math
from statistics import mean

import torch
from tqdm import tqdm

from .Run import Run
from utils import clone_state, clone

class Train(Run):
    def __call__(self, debug):
        if debug:
            return print(repr(self))

        super().prepare()

        best_model = clone_state(self.model)
        best_loss = math.inf
        train_losses, valid_losses = [], []
        for epoch in range(self.epochs):
            self.log('Epoch %i/%i' % (epoch + 1, self.epochs))
            train_loss = self.train()
            train_losses.append(train_loss)

            valid_loss = self.validate()
            valid_losses.append(valid_loss)

            eval_loss = mean(valid_loss)
            if eval_loss <= best_loss:
                best_loss = eval_loss
                best_model = clone_state(self.model)

        self.log('train_loss(%s): %s' \
            % (self.train_dl, [round(x, 4) for x in train_losses]), file=True)
        for valid_dl, losses in zip(self.valid_dls, zip(*valid_losses)):
            self.log('valid_loss(%s): %s' \
                % (valid_dl, [round(x, 4) for x in losses]), file=True)

        # super().terminate(best_model)


    def train(self):
        losses = []
        for data in (t := tqdm(self.train_dl)):
            loss = self.train_batch(data)
            losses.append(loss)
            self.step_lr()
            self.log({'train_loss_%s' % self.train_dl: loss})
            t.set_description('Train loss: %.4f (~%.4f)' % (loss, mean(losses)))
        return mean(losses)


    def validate(self):
        model = clone(self.model)
        valid_loss = []
        for valid_dl in self.valid_dls:
            losses = []
            for data in valid_dl:
                loss = self.validate_batch(model, data)
                losses.append(loss)
            loss_avg = mean(losses)
            self.log({'valid_loss_%s' % valid_dl: loss_avg})
            print('valid_loss(%s): %.4f' % (valid_dl, loss_avg))
            valid_loss.append(loss_avg)
        return valid_loss


    def step_lr(self):
        if self.scheduler:
            self.scheduler.step()
        self.log({'lr': self.get_lr()})


    def get_lr(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        else:
            return self.lr
