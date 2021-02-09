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
            requires=['models', 'train_set', 'valid_sets', 'scale', 'spt_size', 'lr', 'size', 'loss', 
                'update_steps', 'wandb'])

        self.models = []
        for model_fp in opt.models:
            name, autoencoder = create_edsr(model_fp, map_location=torch.device('cpu'))
            model = MAML(autoencoder, lr=opt.lr, first_order=True, allow_nograd=True).to(device)
            self.models.append((name, model))

        self.loss = Loss.get(opt.loss, device)
        self.psnr = PSNR(edge=6 + opt.scale)

        train_set = BasicDataset.preset(opt.train_set, scale=opt.scale, size=opt.size)
        test_set = BasicDataset.preset(opt.valid_sets[0], scale=opt.scale, size=opt.size)
        
        opt.qry_size = opt.spt_size*len(train_set)//len(test_set)
        self.train_dl = DataLoader(train_set, batch_size=opt.spt_size, num_workers=4, shuffle=False)
        self.test_dl = DataLoader(test_set, batch_size=opt.qry_size, num_workers=2, shuffle=False)

        self.summarize(**vars(opt))


    def __call__(self, n_tracked=6):
        super().__call__()

        #steps = [{} for _ in range(self.opt.update_steps + 2)] # extra 2 for step=0 and step=img
        psnrs = {name: 0. for name, _ in self.models}
        for i, (spt, qry) in enumerate(tqdm(zip(self.train_dl, self.test_dl), total=len(self.train_dl))):
            x_spt, y_spt = [d.to(device) for d in spt]
            x_qry, y_qry = [d.to(device) for d in qry]

            # example = [wandb.Image(x_qry[0].cpu(), caption='LR'),
            #            wandb.Image(y_qry[0].cpu(), caption='HR (L1 / PSNR)')]

            for j, (name, model) in enumerate(self.models):
                cloned = model.clone()
                y_qry_hat = cloned(x_qry)
                # psnr = sum(self.psnr(y_hat, y) for y_hat, y in zip(y_qry_hat, y_qry))
                # if i < n_tracked:
                #     steps[0]['model(%i) img_%i' % (j, i)] = psnr.item()
                
                for k in range(self.opt.update_steps):
                    y_spt_hat = cloned(x_spt)
                    loss_spt = self.loss(y_spt_hat, y_spt)
                    cloned.adapt(loss_spt)

                    y_qry_hat = cloned(x_qry)
                    # psnr = self.psnr(y_qry_hat[0], y_qry[0])
                    # if i < n_tracked:
                    #     steps[k]['model(%i) img_%i' % (j, i)] = psnr.item()

                # loss_q = self.loss(y_qry_hat, y_qry)
                psnrs[name] += sum(self.psnr(y_hat, y) for y_hat, y in zip(y_qry_hat, y_qry)).item()
                # if i < n_tracked:
                #     img = wandb.Image(y_qry_hat[0].cpu(),
                #         caption='y_model(%i) (%.4f / %.2f dB)' % (j, loss_q.item(), psnr.item()))
                #     example.append(img)

            # if i < n_tracked:
            #     steps[-1]['img_%i' % i] = example
        
        print('avg PSNR on test set %s' % self.test_dl)
        for i, (name, psnr) in enumerate(psnrs.items()):
            print('%-30s = %.3fdB' % (name, psnr/len(self.test_dl.dataset)))
            
        # for step in steps:
        #     self.log(step)


    def summarize(self, spt_size, qry_size, lr, loss, scale, update_steps, **_):
        models_concat = 'vs'.join(name for name, _ in self.models)
        trainset = str(self.train_dl).replace('_', '-')
        testset = str(self.test_dl).replace('_', '-')

        self._str = '%s[FT%s_s%i_T%s_q%i_u%s]' % (models_concat, trainset, spt_size, testset, qry_size, update_steps)

        self._repr = 'evaluated models : \n   %s \n' % models_concat \
                   + 'finetuning set: \n   %r \n' % self.train_dl \
                   + 'validation set: \n   %r \n' % self.test_dl \
                   + 'scale factor: %s \n' % scale \
                   + 'meta dataset size: #support=%i | #query=%i \n' % (spt_size, qry_size) \
                   + 'update steps: %s \n' % update_steps \
                   + 'learning rate: %s \n' % lr
