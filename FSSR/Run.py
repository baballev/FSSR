import sys

import wandb

from utils import save_state

class Run:
    def __init__(self, mode, requires, options):
        self.mode = mode
        self.opt = options
        self.model = None
        self.debug = options.debug

        if self.debug:
            self.wandb = False
        else:
            self.wandb = options.wandb if options.wandb else mode

        self.require(requires)


    def __call__(self, name=None):
        self.name = name
        if self.debug:
            return print(repr(self))
    
        if self.wandb:
            wandb.init(project=self.wandb, name=str(self), notes=repr(self))
            if self.model is not None:
                wandb.watch(self.model)

    
    def terminate(self, model, loss, epoch):
        fp = '%s_%i_%.3f.pth' % (self.name if self.name else self, epoch, loss)
        save_state(model, fp)
    

    def log(self, payload):
        if (type(payload) is dict and self.wandb):
            wandb.log(payload)
        elif type(payload) is str:
            print(payload)


    def require(self, required):
        errors = []        
        for k in required:
            if hasattr(self.opt, k) and getattr(self.opt, k) is not None:
                continue
            errors.append(' --%s' % k.replace('_', '-'))
        if len(errors) > 0:
            assert False, '--mode=%s requires arguments:\n%s' % (self.mode, '\n'.join(errors))


    def __str__(self):
        return self._str


    def __repr__(self):
        return self._repr
