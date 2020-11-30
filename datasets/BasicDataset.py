from torch.utils.data import Dataset
from torchvision.transforms import Compose

import datasets.transform as t
from .datasets import datasets
from utils import list_images, fetch_image

class BasicDataset(Dataset):
    """Single image dataset."""

    def __init__(self, fp, scale, augment=True, style=False, resize=None):
        self.paths = list_images(datasets[fp] if fp in datasets else fp)
        self.resize = resize
        self.scale = scale
        self.augment = augment
        self.style = style

    def __getitem__(self, index):
        img = fetch_image(self.paths[index])
        resized, scaled = t.get_sizes(*self.resize, self.scale)

        pipeline = []
        if self.augment:
            pipeline.append(t.augment(self.augment))
        if self.style:
            pipeline.append(t.style_filter())

        base = Compose(pipeline)(img)
        x_resize, y_resize = t.resize(scaled), t.resize(resized)
        x, y = x_resize(base), y_resize(base)
        return x, y

    def __len__(self):
        return len(self.paths)
