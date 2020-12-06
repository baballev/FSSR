import wandb

class Run:
    def __init__(self, wandb):
        self.wandb = wandb

    def pre_call(self, name):
        self.logs = Logger(name + '.logs')
        if self.wandb:
            wandb.init(project='tester!', name=name, notes=repr(self))
            wandb.watch(self.model)

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._repr
