import torch
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
            requires=['train_set', 'clusters', 'load', 'scale', 'shots', 'nb_tasks', 'lr', 'meta_lr',
                'size', 'loss', 'n_resblocks', 'n_feats', 'lr_annealing', 'weight_decay', 'epochs', 
                'update_steps', 'update_test_steps', 'first_order', 'timesteps', 'wandb'])

        model = EDSR(opt.n_resblocks, opt.n_feats, opt.scale, res_scale=0.1)
        if opt.load:
            load_state(model, opt.load)

        self.model = MAML(model, lr=opt.meta_lr, first_order=opt.first_order, allow_nograd=True).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.loss = Loss.get(opt.loss, device)

        train_clusters, valid_clusters = get_clusters(opt.clusters, split=0.1, shuffle=False)
        
        #train_set = TaskDataset.preset('DIV2K_train#AUGMENTOR#STYLE', scale=opt.scale, size=opt.size, shots=opt.shots)
        train_set = ClusterDataset.preset(opt.train_set, clusters=train_clusters, scale=opt.scale, 
            size=opt.size, shots=opt.shots)
        self.train_dl = DataLoader(train_set, batch_size=opt.nb_tasks, shuffle=True, num_workers=6)

        #valid_set = TaskDataset.preset('DIV2K_valid', scale=opt.scale, size=opt.size, shots=opt.shots)
        valid_set = ClusterDataset.preset(opt.train_set, clusters=valid_clusters, scale=opt.scale, 
            augment=False, size=opt.size, shots=opt.shots)
        self.valid_dls = [DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=2)]

        if opt.lr_annealing:
            opt.lr_annealing = int(opt.lr_annealing*len(self.train_dl))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, opt.lr_annealing)

        self.epochs = opt.epochs if not opt.timesteps else opt.timesteps//len(self.train_dl)
        self.lr = opt.lr
        self.summarize(**vars(opt))


    def train_batch(self, batch):
        x_spt, y_spt, x_qry, y_qry = [v.to(device) for v in batch]

        loss_q = 0
        for i in range(x_spt.shape[0]):
            cloned = self.model.clone()
            for k in range(self.opt.update_steps):
                y_spt_hat = cloned(x_spt[i])
                loss_spt = self.loss(y_spt_hat, y_spt[i])
                cloned.adapt(loss_spt)

            y_qry_hat = cloned(x_qry[i])
            loss_qry = self.loss(y_qry_hat, y_qry[i])
            loss_q += loss_qry
        loss_q /= self.opt.nb_tasks
       
        self.optim.zero_grad()
        loss_q.backward()
        self.optim.step()
        return loss_q.item()


    def validate_batch(self, model, batch):
        x_spt, y_spt, x_qry, y_qry = [v.to(device) for v in batch]
        assert x_spt.shape[0] == 1, 'Can only be one task per batch on validation'

        cloned = model.clone()
        for k in range(self.opt.update_test_steps):
            y_spt_hat = cloned(x_spt[0])
            loss_spt = self.loss(y_spt_hat, y_spt[0])
            cloned.adapt(loss_spt)

        # query only after all steps have been completed
        y_qry_hat = cloned(x_qry[0]) 
        loss_q = self.loss(y_qry_hat, y_qry[0])
        return loss_q.item()


    def summarize(self, clusters, load, scale, shots, nb_tasks, lr, meta_lr, size, loss, n_resblocks,
        n_feats, lr_annealing, weight_decay, update_steps, update_test_steps, first_order, **_):

        prefix = load.split('.pt')[0] if type(load) is str else ''
        model = 'r%if%ix%i' % (n_resblocks, n_feats, scale)
        dataset = str(self.train_dl).replace('_', '-')

        self._str = '%s%s[%s_%s_lr%s%s_mlr%s_t%s_s%s_u%i]' % (prefix, model, self.mode, dataset, 
            ('%.e' % lr).replace('-0', ''), ('d' if lr_annealing else ''), 
            ('%.e' % meta_lr).replace('-0', ''), nb_tasks, shots, update_steps)

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
                   + 'loss function: %s \n' % loss \
                   + 'tasks per update: %i \n' % nb_tasks \
                   + 'number of shots: %i \n' % shots \
                   + 'updates steps (train): %i \n' % update_steps \
                   + 'update steps (validation): %i \n' % update_test_steps \
                   + 'epochs: %i ' % self.epochs \
                   + '(%i steps)' % (self.epochs*len(self.train_dl))
