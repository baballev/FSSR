import random
import os

import torch
from torchvision.datasets.folder import default_loader

import dataset.transform as t
from .datasets import datasets
from .DatasetWrapper import DatasetWrapper
from utils import list_images, fetch_image

class ClusterDataset(DatasetWrapper):
    """Style-based task segmentation of the dataset."""
    style_params = {'b': 0.3, 'c': 0.5, 's': 0.3, 'h': 0.4}

    def __init__(self, fp, clusters, scale, size, spt_size, qry_size, augment=False, style=False, 
        random=True, loader=default_loader, strict=False):
        
        self.fp = datasets[fp] if fp in datasets else fp
        self.loader = lambda x: loader(os.path.join(self.fp, x))

        self.spt_size, self.qry_size = spt_size, qry_size
        
        self.clusters = self.check_clusters(clusters, strict)
        self.style = style
        self.random = random
        self.augment_name = augment
        self.sizes = t.get_sizes(size, scale)    
        self.augment = t.augment(augment, self.sizes[0])
        self.scale = t.resize(self.sizes[1])

    def check_clusters(self, clusters, strict):
        sizes = [len(x) >= self.shots + 1 for x in clusters]
    
        if strict:
            assert all(sizes)

        return [x for x, big_enough in zip(clusters, sizes) if big_enough]
        

    def __getitem__(self, index):
        cluster = list(self.clusters[index])
        if self.random:
            support = random.sample(cluster[:-1], self.shots)
        else:
            support = cluster[:self.shots]
        
        samples = support + cluster[-1:]
        imgs = [self.loader(fp) for fp in samples]

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
