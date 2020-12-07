import torch
from tqdm import tqdm

from .Run import Run
from utils import load_state
from model import MAML, EDSR, Loss
from dataset import BasicDataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Test(Run):
    def __init__(self, model_fps, valid_fps, scale, shots, lr, size, loss, wandb):
        super(Test, self).__init__(wandb)

        models = []
        for model_fp in model_fps:
            autoencoder = EDSR(n_resblocks=16, n_feats=64, scale=scale, res_scale=0.1)
            load_state(autoencoder, model_fp)
            model = MAML(autoencoder, lr=lr, first_order=True, allow_nograd=True).to(device)
            models.append(model)

        self.loss = Loss.get(loss, device)
        self.model_names  = [m.split('.pt')[0] for m in model_fps]

        test_set = BasicDataset.preset(test_fp, scale=scale, size=size)
        self.test_dl = DataLoader(test_set, batch_size=shots, num_workers=4, shuffle=True)

        self.summarize(shots, lr, loss)
        print(str(self))
        print(repr(self))

    def __call__(self, update_steps):
        super().prepare('%s_e%s' % (self, update_steps))

        losses = {name: [] for name in self.model_names}
        for data in tqdm(self.test_dl):
            x, y = [d.to(device) for d in data]
            y_spt, y_qry = y[:-1], y[-1]
            x_spt, x_qry = x[:-1], x[-1]

            for model, name in zip(self.models, self.model_names):
                # x_spt, y_spt, x_qry, y_qry
                import pdb; pdb.set_trace()
                # losses[name].append(loss)
        # print(losses, file=logs)


    def summarize(self, shots, lr, loss):
        self._str = construct_name(name='vs'.join(self.model_names), bs=shots, action='test')

        self._repr = 'evaluated models : \n' \
                   + ''.join(['    %s \n' % m for m in self.model_names]) \
                   + 'valid sets: \n' \
                   +  ''.join(['   %s \n' % repr(s) for s in self.valid_dls]) \
                   + 'scale factor: %s \n' % scale \
                   + 'number of shots: %s \n' % shots \
                   + 'learning rate: %s \n' % lr \
                   + 'loss function: %s \n' % loss
