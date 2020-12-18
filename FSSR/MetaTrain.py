import torch
import torch.optim as optim

from .Train import Train
from utils import load_state
from model import MAML, EDSR, Loss
from dataset import ClusterDataset, DataLoader, get_clusters

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MetaTrain(Train):
    def __init__(self, dataset_fp, clusters_fp, load=None, scale=2, shots=10, nb_tasks=1, lr=0.001,
        meta_lr=0.0001, size=(256, 512), loss='L1', n_resblocks=16, n_feats=64, lr_annealing=None,
        wandb=False):

        super(MetaTrain, self).__init__('meta!' if wandb else None)

        model = EDSR(n_resblocks, n_feats, scale, res_scale=0.1)
        if load:
            load_state(model, load)

        self.model = MAML(model, lr=meta_lr, first_order=True, allow_nograd=True).to(device)
        self.optim = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.0001)
        self.loss = Loss.get(loss, device)
        self.nb_tasks = nb_tasks

        train_clusters, valid_clusters = get_clusters(clusters_fp, split=0.1, shuffle=False)

        train_set = ClusterDataset.preset(dataset_fp, clusters=train_clusters, scale=scale, size=size, shots=shots)
        self.train_dl = DataLoader(train_set, batch_size=nb_tasks, shuffle=True, num_workers=4, pin_memory=True)
        valid_set = ClusterDataset.preset(dataset_fp, clusters=valid_clusters, augment=False, scale=scale, size=size, shots=shots)
        self.valid_dls = [DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)]

        if lr_annealing:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, lr_annealing*len(self.train_dl))

        self.summarize(load, scale, lr, meta_lr, shots, loss, n_resblocks, n_feats)


    def __call__(self, epochs, update_steps, update_test_steps):
        super().__call__(
            epochs=epochs,
            update_steps=update_steps,
            update_test_steps=update_test_steps,
            name='%s_e%i_u%i]' % (self, epochs, update_steps))


    def train_batch(self, batch, update_steps, **_):
        x_spt, y_spt, x_qry, y_qry = [v.to(device) for v in batch]

        loss_q = 0
        for i in range(self.nb_tasks):
            cloned = self.model.clone()
            for k in range(update_steps):
                y_spt_hat = cloned(x_spt[i])
                loss_spt = self.loss(y_spt_hat, y_spt[i])
                cloned.adapt(loss_spt)

                y_qry_hat = cloned(x_qry[i])
            loss_qry = self.loss(y_qry_hat, y_qry[i])
            loss_q += loss_qry
        loss_q /= self.nb_tasks

        self.optim.zero_grad()
        loss_q.backward()
        self.optim.step()
        return loss_q.item()


    def validate_batch(self, model, batch, update_test_steps, **_):
        x_spt, y_spt, x_qry, y_qry = [v.to(device) for v in batch]
        assert x_spt.shape[0] == 1, 'Can only be one task per batch on validation'

        cloned = model.clone()
        for k in range(update_test_steps):
            y_spt_hat = cloned(x_spt[0])
            loss_spt = self.loss(y_spt_hat, y_spt[0])
            cloned.adapt(loss_spt)

            y_qry_hat = cloned(x_qry[0])
            loss_q = self.loss(y_qry_hat, y_qry[0]) # for every loop, ain't leakin?
        return loss_q.item()


    def summarize(self, load, scale, lr, meta_lr, shots, loss, n_resblocks, n_feats):
        prefix = '' if load is None else load.split('.pt')[0]
        model = 'EDSR-r%if%ix%i' % (n_resblocks, n_feats, scale)
        dataset = str(self.train_dl).replace('_', '-')

        self._str = '%s%s[%s_%s_t%s_s%s' % (prefix, model, 'meta', dataset, self.nb_tasks, shots)

        self._repr = 'train set: \n   %s \n' % repr(self.train_dl) \
                   + 'valid sets: \n' \
                   +  ''.join(['   %s \n' % repr(s) for s in self.valid_dls]) \
                   + 'finetuning: %s \n' % ('from %s' % load if load else 'False') \
                   + 'scale factor: %i \n' % scale \
                   + 'tasks per update: %i \n' % self.nb_tasks \
                   + 'number of shots: %i \n' % shots \
                   + 'learning rate: %s \n' % lr \
                   + 'meta-learning rate: %s \n' % meta_lr \
                   + 'loss function: %s \n' % loss
