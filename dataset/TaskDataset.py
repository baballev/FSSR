import random

import torch

import dataset.transform as t
from .datasets import datasets
from .DatasetWrapper import DatasetWrapper
from utils import list_images, fetch_image

class TaskDataset(DatasetWrapper):
    """Style-based task segmentation of the dataset."""
    style_params = {'b': 0.4, 'c': 0.4, 's': 0.4, 'h': 0.2}

    def __init__(self, fp, scale, size, shots, augment=False, style=False, shuffle=False):
        self.fp = fp
        self.sizes = t.get_sizes(size, scale)
        self.style = style
        self.augment_name = augment
        self.augment = t.augment(augment, self.sizes[0])
        self.scale = t.resize(self.sizes[1])

        if shuffle:
            random.shuffle(self.paths)
        paths = list_images(datasets[fp] if fp in datasets else fp) # remove self.
        task_size = shots + 1
        n_tasks = len(paths) // task_size
        self.tasks = [paths[i: i + task_size] for i in range(0, n_tasks*task_size, task_size)]
        
    def __getitem__(self, index):
        imgs = [fetch_image(fp) for fp in self.tasks[index]]

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
        return len(self.tasks)

