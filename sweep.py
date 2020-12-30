"""Train wrapper script for Wandb sweeps."""

from argparse import ArgumentParser, Namespace

import wandb

from FSSR import MetaTrain

parser = ArgumentParser()
parser.add_argument('sweep_id', type=str, help='Wandb sweep id.') 
args = parser.parse_args()

def train():
    trial = wandb.init()
    
    config = dict(trial.config)
    config['size'] = [config['size']]*2
    
    opt = Namespace(**config)
    MetaTrain(opt)(name=trial.name)

#opt={'clusters': 'DIV2K_160clusters', 'debug': False, 'epochs': 0, 'first_order': True, 'load': False, 'loss': 'L1', 'lr': 0.000792658750105796, 'lr_annealing': 0, 'meta_lr': 0.00019211474409957373, 'n_feats': 64, 'n_resblocks': 16, 'nb_tasks': 1, 'scale': 2, 'shots': 1, 'size': [248, 248], 'timesteps': 10000, 'train_set': 'DIV2K_all#AUGMENTOR', 'update_steps': 10, 'update_test_steps': 10, 'wandb': False, 'weight_decay': 1e-06}
#opt['debug'] = True

#opt = Namespace(**opt)
#MetaTrain(opt)()
wandb.agent(args.sweep_id, train, project='FSSR')
