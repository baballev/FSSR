import torch
import torch.optim as optim

from .Train import Train
from utils import load_state
from model import EDSR, Loss
from dataset import BasicDataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VanillaTrain(Train):
    def __init__(self, opt):
        super(VanillaTrain, self).__init__(mode='vanilla', options=opt,
            requires=['train_set', 'valid_sets', 'load', 'scale', 'batch_size', 'lr', 'size', 
                'loss', 'n_resblocks', 'n_feats', 'epochs', 'wandb'])

        self.model = EDSR(opt.n_resblocks, opt.n_feats, opt.scale, res_scale=0.1).to(device)
        if opt.load:
            load_state(self.model, opt.load)

        self.optim = optim.Adam(self.model.parameters(), lr=opt.lr)#, amsgrad=True)
        # optim should be halved at every 2x10^5 minibatch updates
        self.loss = Loss.get(opt.loss, device)

        train_set = BasicDataset.preset(opt.train_set, scale=opt.scale, size=opt.size)
        self.train_dl = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=6)

        self.valid_dls = []
        for valid in opt.valid_sets:
            valid_set = BasicDataset.preset(valid, scale=opt.scale, size=opt.size)
            valid_dl = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
            self.valid_dls.append(valid_dl)

        self.epochs = opt.epochs if not opt.timesteps else opt.timesteps//len(self.train_dl)        
        self.lr = opt.lr
        self.summarize(**vars(opt))


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


    def summarize(self, load, scale, batch_size, lr, loss, n_resblocks, n_feats, epochs, **_):
        prefix = load.split('.pt')[0] if type(load) is str else ''
        model = 'r%if%ix%i' % (n_resblocks, n_feats, scale)
        dataset = str(self.train_dl).replace('_', '-')

        self._str = '%s%s[%s_%s_bs%s_e%i]' % (prefix, model, self.mode, dataset, batch_size, epochs)

        self._repr = 'run name: %s \n' % self \
                   + 'train set: \n   %s \n' % repr(self.train_dl) \
                   + 'valid sets: \n' \
                   +  ''.join(['   %s \n' % repr(s) for s in self.valid_dls]) \
                   + 'finetuning: %s \n' % (load if load else 'False') \
                   + 'scale factor: %s \n' % scale \
                   + 'batch size: %s \n' % batch_size \
                   + 'learning rate: %s \n' % lr \
                   + 'loss function: %s \n' % loss \
                   + 'epochs: %i' % epochs \
                   + '(%i steps)' % (epochs*len(self.train_dl))
