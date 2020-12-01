import random

import torch

import datasets.transform as t
from .datasets import datasets
from utils import list_images, fetch_image

class TaskDataset(torch.utils.data.Dataset):
    """Style-based task segmentation of the dataset."""

    def __init__(self, fp, shots, scale, augment=True, style=True, resize=None):
        self.paths = list_images(datasets[fp] if fp in datasets else fp)
        self.resize = resize
        self.scale = scale
        self.shots = shots
        self.style = style
        self.augment = t.augment(augment) if augment else None

    def __getitem__(self, index):
        img_paths = random.sample(self.paths, self.shots + 1)
        imgs = [fetch_image(path) for path in img_paths]
        resized, scaled = t.get_sizes(*self.resize, self.scale)

        pipeline = t.Pipeline()
        if self.augment:
            pipeline.add(self.augment)
        if self.style:
            pipeline.add(t.style_filter())

        bases = [pipeline(img) for img in imgs]
        y = torch.stack([t.resize(resized)(m) for m in bases])
        x = torch.stack([t.resize(scaled)(m) for m in bases])

        y_spt, y_qry = y[:-1], y[-1]
        x_spt, x_qry = x[:-1], x[-1]
        return x_spt, y_spt, x_qry, y_qry

    def __len__(self):
        return int(len(self.paths)//self.shots)

