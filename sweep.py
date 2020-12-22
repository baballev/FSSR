"""Train wrapper script for Wandb sweeps."""

from argparse import ArgumentParser, Namespace

import wandb

from FSSR import MetaTrain

parser = ArgumentParser()
parser.add_argument('sweep_id', type=str, help='Wandb sweep id.') 
sweep_id = parser.parse_args().sweep_id

def train():
    run = wandb.init()
    config = dict(run.config)
    config['update_test_steps'] = config['update_steps']
    config['size'] = [config['size']]*2
    opt = Namespace(**config)
    MetaTrain(opt)()

wandb.agent(sweep_id, train, project='FSSR')
