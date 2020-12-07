import random

import torch

import dataset.transform as t
from .datasets import datasets
from .DatasetWrapper import DatasetWrapper
from utils import list_images, fetch_image

class TaskDataset(DatasetWrapper):
    """Style-based task segmentation of the dataset."""
    style_params = {'b': 0.2, 'c': 0.2, 's': 0.2, 'h': 0.1}

    def __init__(self, fp, scale, size, shots, augment=False, style=False):
        self.fp = fp
        self.paths = list_images(datasets[fp] if fp in datasets else fp)
        self.sizes = t.get_sizes(size, scale)
        self.style = style
        self.shots = shots
        self.augment_name = augment
        self.augment = t.augment(augment, self.sizes[0])
        self.scale = t.resize(self.sizes[1])

    def __getitem__(self, index):
        img_paths = random.sample(self.paths, self.shots + 1)
        imgs = [fetch_image(path) for path in img_paths]

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
        return int(len(self.paths)//self.shots)

