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
            assert hasattr(opt, arg) and getattr(opt, arg), \
                'argument --{} required to run --mode={}'.format(arg.replace('_', '-'), opt.mode)
    return require

def summarize_args(opt, verbose):
    def summarize(*args):
        for arg in args:
            print(verbose[arg](getattr(opt, arg))) if arg in verbose and hasattr(opt, arg) else 0
    return summarize

def construct_name(name=None, load=None, dataset='?', bs=0, action='?'):
    if load:
        fp = os.path.normpath(load)
        prefix = fp.split(os.sep)[-1].split('.pt')[0]
    elif name:
        prefix = name
    else:
        prefix = 'Unknown'
    return '%s[%s_%s_bs%i' % (prefix, action, dataset.replace('_', '-'), bs)

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

def load_state(model, fp):
    """Load weights located in file fp onto the model."""
    weights = torch.load(fp)
    model.load_state_dict(weights)
