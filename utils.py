import os, sys, imghdr, copy

import torch
from PIL import Image

class Logger(object):
    def __init__(self, fp):
        self.file = open(fp, 'w', buffering=1)

    def write(self, str):
        sys.stdout.write(str)
        self.file.write(str)

    def flush(self):
        pass

    def __del__(self):
        self.file.close()

def require_args(opt):
    def require(*args):
        for arg in args:
            assert hasattr(opt, arg) and getattr(opt, arg) is not None, \
                'argument --{} required to run --mode={}'.format(arg.replace('_', '-'), opt.mode)
    return require

def is_image(path):
    return imghdr.what(path) == 'jpeg' or imghdr.what(path) == 'png'

def list_directory_files(path, policy):
    return [os.path.join(path, f) for f in sorted(os.listdir(path)) if policy(os.path.join(path, f))]

def list_images(path):
    return list_directory_files(path, is_image)

def fetch_image(path):
    return Image.open(path).convert('RGB')

def save_state(state_dict, fp):
    """Dumps the model's state to file fp."""
    torch.save(state_dict, fp)

def clone(model):
    """Deep copy of the entire module."""
    return copy.deepcopy(model)

def clone_state(model):
    """Deep copy of the model's state."""
    return copy.deepcopy(model.state_dict())

def load_state(model, fp, **kwargs):
    """Load weights located in file fp onto the model."""
    weights = torch.load(fp, **kwargs)
    model.load_state_dict(weights)
