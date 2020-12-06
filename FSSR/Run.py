import wandb

from utils import Logger

class Run:
    project = 'tester!'

    def __init__(self, wandb):
        self.wandb = wandb

    def pre_call(self, name):
        self.out = name + '.pth'
        self.logs = Logger(name + '.logs')
        if self.wandb:
            wandb.init(project=self.project, name=name, notes=repr(self))
            wandb.watch(self.model)

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._repr
