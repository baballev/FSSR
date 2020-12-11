import torch
import torch.optim as optim

from .Train import Train
from utils import load_state
from model import EDSR, Loss
from dataset import BasicDataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VanillaTrain(Train):
    def __init__(self, train_fp, valid_fps, load=None, scale=2, bs=1, lr=0.0001, size=(256, 512),
        loss='L1', n_resblocks=16, n_feats=64, wandb=False):
        super(VanillaTrain, self).__init__(wandb)

        self.model = EDSR(n_resblocks, n_feats, scale, res_scale=0.1).to(device)
        if load:
            load_state(self.model, load)

        self.optim = optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.loss = Loss.get(loss, device)

        train_set = BasicDataset.preset(train_fp, scale=scale, size=size)
        self.train_dl = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=6, pin_memory=True)

        self.valid_dls = []
        for valid_fp in valid_fps:
            valid_set = BasicDataset.preset(valid_fp, scale=scale, size=size)
            valid_dl = DataLoader(valid_set, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
            self.valid_dls.append(valid_dl)

        self.summarize(load, scale, bs, lr, loss, n_resblocks, n_feats)


    def __call__(self, epochs):
        super().__call__(name='%s_e%i]' % (self, epochs), epochs=epochs)


    def train_batch(self, batch):
        x, y = [v.to(device) for v in batch]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()


    @torch.no_grad()
    def validate_batch(self, model, batch):
        x, y = [d.to(device) for d in batch]
        y_hat = model(x)
        loss = self.loss(y_hat, y)
        return loss.item()


    def summarize(self, load, scale, bs, lr, loss, n_resblocks, n_feats):
        prefix = '' if load is None else load.split('.pt')[0]
        model = 'EDSR-r%if%ix%i' % (n_resblocks, n_feats, scale)
        dataset = str(self.train_dl).replace('_', '-')

        self._str = '%s%s[%s_%s_bs%s' % (prefix, model, 'vanilla', dataset, bs)

        self._repr = 'train set: \n   %s \n' % repr(self.train_dl) \
                   + 'valid sets: \n' \
                   +  ''.join(['   %s \n' % repr(s) for s in self.valid_dls]) \
                   + 'finetuning: %s \n' % ('from %s' % load if load else 'False') \
                   + 'scale factor: %s \n' % scale \
                   + 'batch size: %s \n' % bs \
                   + 'learning rate: %s \n' % lr \
                   + 'loss function: %s \n' % loss
