import torch
import wandb
from tqdm import tqdm

from .Run import Run
from utils import load_state
from metrics import PSNR
from model import MAML, EDSR, Loss, create_edsr
from dataset import BasicDataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Test(Run):
    def __init__(self, opt):
        super(Test, self).__init__(mode='eval', options=opt,
            requires=['models', 'valid_sets', 'scale', 'shots', 'lr', 'size', 'loss', 
                'update_steps', 'wandb'])

        self.models = []
        self.model_names = []
        for model_fp in opt.models:
            name, autoencoder = create_edsr(model_fp, map_location=torch.device('cpu'))
            model = MAML(autoencoder, lr=opt.lr, first_order=True, allow_nograd=True).to(device)
            self.models.append(model)
            self.model_names.append(name)

        self.loss = Loss.get(opt.loss, device)
        self.psnr = PSNR(edge=6 + opt.scale)

        test_set = BasicDataset.preset(opt.valid_sets[0], scale=opt.scale, size=opt.size)
        self.test_dl = DataLoader(test_set, batch_size=opt.shots + 1, num_workers=4, shuffle=False)

        self.summarize(**vars(opt))


    def __call__(self, debug):
        super().__call__(debug)

        psnrs = {name: 0. for name in self.model_names}
        for j, data in enumerate(self.test_dl):
            x, y = [d.to(device) for d in data]
            y_spt, y_qry = y[:-1], y[-1:]
            x_spt, x_qry = x[:-1], x[-1:]

            example = [wandb.Image(x_qry[0].cpu(), caption='LR'),
                       wandb.Image(y_qry[0].cpu(), caption='HR (L1 / PSNR)')]

            for i, (model, name) in enumerate(zip(self.models, self.model_names)):
                cloned = model.clone()
                for k in range(self.opt.update_steps):
                    y_spt_hat = cloned(x_spt)
                    loss_spt = self.loss(y_spt_hat, y_spt)
                    cloned.adapt(loss_spt)

                y_qry_hat = cloned(x_qry)
                loss_q = self.loss(y_qry_hat, y_qry)
                psnr = self.psnr(y_qry_hat[0], y_qry[0])
               
                self.log({name: loss_q.item()})
                psnrs[name] += psnr.item()
                if j < 6:
                    img = wandb.Image(y_qry_hat[0].cpu(),
                        caption='y_model(%i) (%.4f / %.2f dB)' % (i, loss_q.item(), psnr.item()))
                    example.append(img)

            if j < 6:
                self.log({'img_%i' % j: example})
        
        for i, (model, psnr) in enumerate(psnrs.items()):
            print('model(%i) avg PSNR = %.2f dB \n  %s' % (i, psnr/len(self.test_dl), model))


    def summarize(self, shots, lr, loss, scale, update_steps, **_):
        models_concat = 'vs'.join(self.model_names)
        dataset = str(self.test_dl).replace('_', '-')

        self._str = '%s[%s_%s_s%s_u%s]' % (models_concat, self.mode, dataset, shots, update_steps)

        self._repr = 'evaluated models : \n' \
                   + 'validation set: \n' \
                   + '   %s \n' % repr(self.test_dl) \
                   + 'scale factor: %s \n' % scale \
                   + 'number of shots: %s \n' % shots \
                   + 'update steps: %s \n' % update_steps \
                   + 'learning rate: %s \n' % lr
