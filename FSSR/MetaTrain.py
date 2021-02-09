import math

import torch
import torch.nn as nn
import torch.optim as optim
 
import wandb
from .Train import Train
from utils import load_state
from model import MAML, EDSR, Loss
from dataset import ClusterDataset, DataLoader, get_clusters, TaskDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MetaTrain(Train):
    def __init__(self, opt):
        super(MetaTrain, self).__init__(mode='meta', options=opt,
            requires=['train_set', 'clusters', 'load', 'scale', 'spt_size', 'qry_size', 'nb_tasks', 
                'lr', 'meta_lr', 'size', 'loss', 'n_resblocks', 'n_feats', 'lr_annealing', 
                'weight_decay', 'epochs', 'update_steps', 'update_test_steps', 'first_order', 
                'clip_norm', 'timesteps', 'wandb'])

        model = EDSR(opt.n_resblocks, opt.n_feats, opt.scale, res_scale=0.1)
        if opt.load:
            load_state(model, opt.load)

        self.model = MAML(model, lr=opt.meta_lr, first_order=opt.first_order, allow_nograd=True).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.loss = Loss.get(opt.loss, device)

        train_clusters, valid_clusters = get_clusters(opt.clusters, split=0.1, shuffle=False)
        
        train_set = ClusterDataset.preset(opt.train_set, clusters=train_clusters, scale=opt.scale, 
            size=opt.size, spt_size=opt.spt_size, qry_size=opt.qry_size, random=True)
        self.train_dl = DataLoader(train_set, batch_size=opt.nb_tasks, shuffle=True, num_workers=4)

        valid_set = ClusterDataset.preset(opt.train_set, clusters=valid_clusters, scale=opt.scale, 
            augment=False, style=False, size=opt.size, spt_size=opt.spt_size, qry_size=opt.qry_size, random=False)
        self.valid_dls = [DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=2)]

        if opt.lr_annealing:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, 
                T_max=opt.lr_annealing, eta_min=1e-5)

        self.epochs = opt.epochs if not opt.timesteps else opt.timesteps//len(self.train_dl)
        self.lr = opt.lr
        self.update_steps = opt.update_steps
        self.update_test_steps = opt.update_test_steps
        self.summarize(**vars(opt))


    def forward(self, model, x, y):
        y_hat = model(x)
        loss = self.loss(y_hat, y)
        return loss


    def train_batch(self, batch):
        batch = [v.to(device) for v in batch]

        loss_q = 0.
        for x_spt, y_spt, x_qry, y_qry in zip(*batch):
            model = self.model.clone()
            
            qry_losses = []
            for k in range(self.update_steps):
                loss_spt = self.forward(model, x_spt, y_spt)
                model.adapt(loss_spt)

                loss_qry = self.forward(model, x_qry, y_qry)
                qry_losses.append(loss_qry)
                
                if loss_qry > 1 or math.isnan(loss_qry): # debug
                    import pdb; pdb.set_trace()

            if self.update_steps:
                loss_qry = torch.stack(qry_losses).mean()
            else:
                loss_qry = self.forward(model, x_qry, y_qry)
            
            loss_q += loss_qry
        loss_q /= self.opt.nb_tasks
       
        self.optim.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.clip_norm)
        self.optim.step()
        return loss_q.item()


    def validate_batch(self, model, batch):
        x_spt, y_spt, x_qry, y_qry = [v.to(device) for v in batch]
        assert x_spt.shape[0] == 1, 'Can only be one task per batch on validation'

        cloned = model.clone()
        for k in range(self.update_test_steps):
            y_spt_hat = cloned(x_spt[0])
            loss_spt = self.loss(y_spt_hat, y_spt[0])
            cloned.adapt(loss_spt)

        # query only after all steps have been completed
        y_qry_hat = cloned(x_qry[0]) 
        loss_q = self.loss(y_qry_hat, y_qry[0])
        return loss_q.item()


    def summarize(self, clusters, load, scale, spt_size, qry_size, nb_tasks, lr, meta_lr, size, 
        loss, n_resblocks, n_feats, lr_annealing, weight_decay, update_steps, update_test_steps, 
        first_order, clip_norm, **_):

        prefix = load.split('.pt')[0] if type(load) is str else ''
        model = 'r%if%ix%i' % (n_resblocks, n_feats, scale)
        dataset = str(self.train_dl).replace('_', '-')

        self._str = '%s%s[%s_%s_lr%s%s_mlr%s_t%s_s%i_q%i_u%i]' % (prefix, model, self.mode, dataset, 
            ('%.e' % lr).replace('-0', ''), ('d' if lr_annealing else ''), 
            ('%.e' % meta_lr).replace('-0', ''), nb_tasks, spt_size, qry_size, update_steps)

        nb_calls = len(self.train_dl)*nb_tasks*(spt_size + qry_size)*update_steps
        nb_accesses = len(self.train_dl)*nb_tasks*(spt_size + qry_size)
        self._repr = 'run name: %s \n' % self \
                   + 'train set: \n   %r \n' % self.train_dl \
                   + 'number of batches: %i \n' % len(self.train_dl) \
                   + 'valid sets: \n' \
                   +  ''.join(['   %r \n' % s for s in self.valid_dls]) \
                   + 'finetuning: %s \n' % (load if load else 'False') \
                   + 'scale factor: %i \n' % scale \
                   + '1st order? (FOMAML): %r \n' % first_order \
                   + 'lr (outer loop): %s \n' % lr \
                   + 'lr decay: %s \n' % (('anneals every %i steps' % lr_annealing) if lr_annealing else 'No') \
                   + 'meta-lr (inner-loop): %s \n' % meta_lr \
                   + 'weight decay: %s \n' % ('none' if weight_decay == 0 else weight_decay) \
                   + 'clip gradient norm: %f \n' % clip_norm \
                   + 'loss function: %s \n' % loss \
                   + 'tasks per update: %i \n' % nb_tasks \
                   + 'meta dataset size: #support=%i | #query=%i \n' % (spt_size, qry_size) \
                   + 'updates steps (train): %i \n' % update_steps \
                   + 'update steps (validation): %i \n' % update_test_steps \
                   + 'epochs: %i ' % self.epochs \
                   + '(%i steps) \n' % (self.epochs*len(self.train_dl)) \
                   + 'budget /epoch: %i forward passes | %i file accesses \n' % (nb_calls, nb_accesses)
