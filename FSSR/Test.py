import torch
import wandb
from tqdm import tqdm

from .Run import Run
from utils import load_state
from model import MAML, EDSR, Loss, create_edsr
from dataset import BasicDataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Test(Run):
    def __init__(self, model_fps, valid_fps, scale, shots, lr, size, loss, wandb):
        super(Test, self).__init__(wandb='test!' if wandb else None)

        self.models = []
        self.model_names = []
        for model_fp in model_fps:
            name, autoencoder = create_edsr(model_fp, map_location=torch.device('cpu'))
            model = MAML(autoencoder, lr=lr, first_order=True, allow_nograd=True).to(device)
            self.models.append(model)
            self.model_names.append(name)

        self.loss = Loss.get(loss, device)
        self.model_names  = [m.split('.pt')[0] for m in model_fps]

        test_fp = valid_fps[0]
        test_set = BasicDataset.preset(test_fp, scale=scale, size=size)
        self.test_dl = DataLoader(test_set, batch_size=shots + 1, num_workers=4, shuffle=True)

        self.summarize(shots, lr, loss, scale)


    @torch.no_grad()
    def __call__(self, update_steps):
        super().prepare('%s_e%s]' % (self, update_steps))

        losses = {name: [] for name in self.model_names}
        for j, data in enumerate(self.test_dl):
            x, y = [d.to(device) for d in data]
            y_spt, y_qry = y[:-1], y[-1:]
            x_spt, x_qry = x[:-1], x[-1:]

            example = [
                wandb.Image(x_qry[0].cpu(), caption='input'),
                wandb.Image(y_qry[0].cpu(), caption='y_true'),
            ]
            for model, name in zip(self.models, self.model_names):
                cloned = model.clone()
                for k in range(update_steps):
                    y_spt_hat = cloned(x_spt)
                    loss_spt = self.loss(y_spt_hat, y_spt)
                    cloned.adapt(loss_spt)
                y_qry_hat = cloned(x_qry)
                loss_q = self.loss(y_qry_hat, y_qry)

                losses[name].append(loss_q.item())
                self.log({name: loss_q.item()})
                example.append(wandb.Image(y_qry_hat[0].cpu(), caption='y_%s:%.4f' % (name, loss_q.item())))
            if j < 4:
                self.log({'img_%i' % j: example})


    def summarize(self, shots, lr, loss, scale):
        self._str = '%s[%s_%s_bs%s' % ('vs'.join(self.model_names), 'test', str(self.test_dl).replace('_', '-'), shots)

        self._repr = 'evaluated models : \n' \
                   + ''.join(['    %s \n' % m for m in self.model_names]) \
                   + 'validation sets: \n' \
                   + 'scale factor: %s \n' % scale \
                   + 'number of shots: %s \n' % shots \
                   + 'learning rate: %s \n' % lr \
                   + 'loss function: %s \n' % loss
