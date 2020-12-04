import os

import dataset.transform as t
from .datasets import datasets
from .DatasetWrapper import DatasetWrapper
from utils import list_images, fetch_image

class BasicDataset(DatasetWrapper):
    """Single image dataset."""
    style_params = {'b': 0.2, 'c': 0.2, 's': 0.2, 'h': 0.1}

    def __init__(self, fp, scale, size, augment=False, style=False):
        self.fp = fp
        self.paths = list_images(datasets[fp] if fp in datasets else fp)
        self.sizes = t.get_sizes(size, scale)
        self.scale = scale
        self.style = style
        self.augment_name = augment
        self.augment = t.augment(augment) if augment else None

    def __getitem__(self, index):
        img = fetch_image(self.paths[index])
        resized, scaled = self.sizes

        p = t.Pipeline()
        if self.augment:
            p.add(self.augment)
        if self.style:
            p.add(t.style_filter(**self.style_params))
        base = p(img)

        x, y = t.resize(scaled)(base), t.resize(resized)(base)
        return x, y

    def __len__(self):
        return len(self.paths)
