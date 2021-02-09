import os, random

import torch
from torchvision.datasets.folder import default_loader

import dataset.transform as t
from .datasets import datasets
from .DatasetWrapper import DatasetWrapper
from utils import list_images, fetch_image


class ClusterDataset(DatasetWrapper):
    """Generic cluster based task segmentation of dataset."""
    style_params = {'b': 0.3, 'c': 0.3, 's': 0.3, 'h': 0.4}

    def __init__(self, fp, clusters, scale, size, spt_size, qry_size, augment=False, style=False, 
        random=True, loader=default_loader, strict=False):
        
        self.fp = datasets[fp] if fp in datasets else fp
        self.loader = lambda x: loader(os.path.join(self.fp, x))

        self.spt_size, self.qry_size = spt_size, qry_size
        self.clusters = self.split_clusters(clusters, strict)
        
        self.style = style
        self.random = random
        self.augment_name = augment
        self.sizes = t.get_sizes(size, scale)    
        self.scale = t.resize(self.sizes[0])
        self.augment = t.augment(augment, self.sizes[1])


    def split_clusters(self, clusters, strict):
        clusters_ = [x for x in clusters if len(x) >= self.spt_size + self.qry_size]
        assert not strict or len(clusters) == len(clusters_)

        meta_splits = []
        for c in clusters_:
            test_size = self.qry_size * len(c) // (self.spt_size + self.qry_size)
            meta_splits.append((c[:-test_size], c[-test_size:]))
        return meta_splits


    def __getitem__(self, index):
        train, test = list(self.clusters[index])

        if self.random:
            support = random.sample(train, self.spt_size)
            query = random.sample(test, self.qry_size)
        else:
            support, query = train[:self.spt_size], test[-self.qry_size:]

        imgs = [self.loader(fp) for fp in support + query]

        p = t.Pipeline(self.augment)
        if self.style:
            p.add(t.style_filter(**self.style_params))
        bases = [p(img) for img in imgs]

        y = torch.stack([t.tensor(m) for m in bases])
        x = torch.stack([self.scale(m) for m in bases])

        y_spt, y_qry = y.split((self.spt_size, self.qry_size))
        x_spt, x_qry = x.split((self.spt_size, self.qry_size))
        return x_spt, y_spt, x_qry, y_qry


    def __len__(self):
        return len(self.clusters)
