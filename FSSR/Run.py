import sys

import wandb

from utils import Logger, save_state

class Run:
    def __init__(self, wandb):
        self.wandb = wandb
        self.model = None

    def prepare(self, name):
        self.out = name + '.pth'

        if self.wandb:
            wandb.init(project=self.wandb, name=name, notes=repr(self))
            if self.model is not None:
                wandb.watch(self.model)

    def terminate(self, best_model):
        save_state(best_model, self.out)

    def log(self, payload, file=False):
        if (type(payload) is dict and self.wandb):
            wandb.log(payload)
        else:
            print(payload)

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._repr
