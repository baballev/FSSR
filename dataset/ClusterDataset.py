import random
import os

import torch
import numpy as np

import dataset.transform as t
from .datasets import datasets
from .DatasetWrapper import DatasetWrapper
from utils import list_images, fetch_image

class ClusterDataset(DatasetWrapper):
    """Style-based task segmentation of the dataset."""
    style_params = {'b': 0.3, 'c': 0.5, 's': 0.3, 'h': 0.4}

    def __init__(self, fp, clusters, scale, size, shots, augment=False, style=False):
        assert min([len(c) for c in clusters]) >= shots + 1
        
        self.clusters = clusters
        self.fp = datasets[fp] if fp in datasets else fp
        self.shots = shots
        self.style = style
        self.augment_name = augment
        self.sizes = t.get_sizes(size, scale)    
        self.augment = t.augment(augment, self.sizes[0])
        self.scale = t.resize(self.sizes[1])

    def __getitem__(self, index):
        cluster = list(self.clusters[index])
        samples = random.sample(cluster, self.shots + 1)
        imgs = [fetch_image(os.path.join(self.fp, path)) for path in samples]

        p = t.Pipeline(self.augment)
        if self.style:
            p.add(t.style_filter(**self.style_params))
        bases = [p(img) for img in imgs]

        y = torch.stack([t.tensor(m) for m in bases])
        x = torch.stack([self.scale(m) for m in bases])

        y_spt, y_qry = y[:-1], y[-1:]
        x_spt, x_qry = x[:-1], x[-1:]
        return x_spt, y_spt, x_qry, y_qry

    def __len__(self):
        return len(self.clusters)
