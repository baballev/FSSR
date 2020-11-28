import random

import torch
from torchvision.transforms import Compose, ToTensor

from utils import list_images, fetch_image
import datasets.transform as transform
from .datasets import datasets

class TaskDataset(torch.utils.data.Dataset):
    """Style-based task segmentation of the dataset."""

    def __init__(self, fp, shots, scale, augment=True, style=False, resize=None):
        self.paths = list_images(datasets[fp] if fp in datasets else fp)
        self.resize = resize
        self.scale = scale
        self.shots = shots
        self.augment = augment
        self.style = style

    def __getitem__(self, index):
        img_paths = random.sample(self.paths, self.shots + 1)
        imgs = [fetch_image(path) for path in img_paths]
        resized, scaled = transform.get_sizes(*self.resize, self.scale)

        pipeline = []
        if self.augment:
            pipeline.append(transform.augment())
        if self.style:
            pipeline.append(transform.style_filter())
        pipeline += [transform.resize(resized), ToTensor()]
        scale = transform.resize(scaled)

        T = Compose(pipeline)
        y = torch.stack([T(img) for img in imgs])
        x = scale(y)

        y_spt, y_qry = y[:-1], y[-1]
        x_spt, x_qry = x[:-1], x[-1]
        return x_spt, y_spt, x_qry, y_qry

    def __len__(self):
        return int(len(self.paths)//self.shots)

