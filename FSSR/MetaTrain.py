import math
from statistics import mean

import wandb
import torch
from tqdm import tqdm
import torch.optim as optim

from utils import Logger, construct_name, load_state, clone_state, save_state
from model import MAML, EDSR, Loss
from dataset import TaskDataset, BasicDataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MetaTrain:
    def __init__(self, train_fp, valid_fp, load=None, scale=2, shots=10, bs=1,
        lr=0.0001, meta_lr=0.00001, size=(256, 512), loss='L1', n_resblocks=16, n_feats=64):

        model = EDSR(n_resblocks, n_feats, scale)
        if load:
            load_state(model, load)

        self.model = MAML(model, lr=meta_lr).to(device)
        self.optim = optim.SGD(self.model.parameters(), lr=lr)
        self.loss = Loss.get(loss, device)

        train_set = TaskDataset.preset(train_fp, scale, size, shots)
        self.train_dl = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)

        self.valid_dls = []
        # for valid_fp in valid_fps:
        #     valid_set = BasicDataset.preset(valid_fp, scale, size)
        #     valid_dl = DataLoader(valid_set, batch_size=bs, shuffle=True, num_workers=2)
        #     self.valid_dls.append(valid_dl)

        self.summarize(load, scale, bs, lr, meta_lr, shots, loss, n_resblocks, n_feats)


    def __call__(self, epochs, update_steps):
        name = '%s_e%i_u%i].pth' % (self.name, epochs, update_steps)
        wandb.init(project='tester!', name=name, notes=self.repr)
        wandb.watch(self.model)
        self.logs = Logger(name + '.logs')

        best_model = clone_state(self.model)
        best_loss = math.inf

        train_losses, valid_losses = [], []
        for epoch in range(epochs):
            print('Epoch %i/%i' % (epoch + 1, epochs))
            self.train(update_steps)
            # train_loss = self.train(update_steps)
            # train_losses.append(train_loss)

            # valid_loss = self.validate()
            # valid_losses.append(valid_loss)

            # eval_loss = mean(valid_loss)
            # if eval_loss < best_loss:
            #     best_loss = eval_loss
            #     best_model = clone_state(self.model)

        # print('train_loss(%s): %s' \
        #     % (str(train_dl), [round(x, 4) for x in train_losses]), file=self.logs)
        # for valid_dl, losses in zip(self.valid_dls, zip(*valid_losses)):
        #     print('valid_loss(%s): %s' \
        #         % (valid_dl, [round(x, 4) for x in losses]), file=self.logs)
        save_state(best_model, name + '.pth')


    def train(self, update_steps):
        # losses = []
        for data in (t := tqdm(self.train_dl)):
            losses_q = 0
            for i in range(update_steps):
                x_spt, y_spt, x_qry, y_qry = [d.to(device) for d in data]

                cloned = self.model.clone()
                y_spt_hat = cloned(x_spt)
                loss = self.loss(y_spt_hat, y_spt)
                print('support loss (y_spt_hat vs y_spt) = %.4f' % loss)
                cloned.adapt(loss)

                y_qry_hat = cloned(x_qry)
                loss_q = self.loss(y_qry_hat, y_qry)
                print('query loss (y_qry_hat vs y_qry) = %.4f' % loss_q)
                losses_q += loss_q

            self.optim.zero_grad()
            losses_q.backward()
            self.optim.step()

            # losses.append(loss.item())
            # wandb.log({'train_loss_%s' % self.train_dl: loss})
            # t.set_description('Train loss: %.4f (~%.4f)' % (loss, mean(losses)))
        # return mean(losses)


    # @torch.no_grad()
    # def validate(self):
    #     valid_losses = []
    #     for valid_dl in self.valid_dls:
    #         losses = []
    #         for data in valid_dl:
    #             x, y = [d.to(device) for d in data]
    #             y_hat = self.model(x)
    #             loss = self.loss(y_hat, y)
    #             losses.append(loss.item())

    #         valid_loss = mean(losses)
    #         valid_losses.append(valid_loss)
    #         wandb.log({'valid_loss_%s' % valid_dl: valid_loss})
    #         print('valid_loss(%s): %.4f' % (valid_dl, valid_loss))
    #     return valid_losses


    def summarize(self, load, scale, bs, lr, meta_lr, shots, loss, n_resblocks, n_feats):
        self.name = construct_name(name='EDSR-r%if%ix%i' % (scale, n_resblocks, n_feats),
            load=load, dataset=str(self.train_dl), bs=bs, action='meta')

        self.repr = 'train set: \n   %s \n' % repr(self.train_dl) \
                  + 'valid sets: \n' \
                  +  ''.join(['   %s \n' % repr(s) for s in self.valid_dls]) \
                  + 'finetuning: %s \n' % ('from %s' % load if load else 'False') \
                  + 'scale factor: %i \n' % scale \
                  + 'tasks per update: %i \n' % bs \
                  + 'number of shots: %i \n' % shots \
                  + 'learning rate: %s \n' % lr \
                  + 'meta-learning rate: %s \n' % meta_lr \
                  + 'loss function: %s \n' % loss

    def __repr__(self):
        return self.repr
