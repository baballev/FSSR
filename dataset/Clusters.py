from math import ceil

import numpy as np
from .datasets import datasets

def get_clusters(fp, split, shuffle):
    """Splits the clusters contained in the file in two sets."""
    assert 0 < split < 1
    
    clusters = np.load(datasets[fp] if fp in datasets else fp, allow_pickle=True)

    if shuffle:
        np.random.shuffle(clusters)

    idx = ceil((1-split)*len(clusters))
    return clusters[:idx], clusters[idx:]
