import re

from . import EDSR
from utils import load_state


def create_edsr(file, **kwargs):
    """
    Instanciates a EDSR model with the correct config inferred from the file name then loads that
    file weights onto the model.

    Arguments
    ---
    file: string - must contain the config of the model in the following format 'EDSR-r#f#x#' where
    # are, respectively the number of res blocks, the number of feature maps and the scale.
    """
    s = re.search(r'(r([1-9]*)f([1-9]*)x([1-9]*).*)\.pth$', file)
    if s is None:
        raise ValueError('file name does not respect \'EDSR-r#f#x#\' syntax')

    name, r, f, x = s.group(1), s.group(2), s.group(3), s.group(4)
    model = EDSR(n_resblocks=int(r), n_feats=int(f), scale=int(x), res_scale=0.1)
    load_state(model, file, **kwargs)

    return name, model

