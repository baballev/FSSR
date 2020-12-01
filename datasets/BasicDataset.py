from torch.utils.data import Dataset
import datasets.transform as t
from .datasets import datasets
from utils import list_images, fetch_image

class BasicDataset(Dataset):
    """Single image dataset."""

    def __init__(self, fp, scale, augment=True, style=False, resize=None):
        self.paths = list_images(datasets[fp] if fp in datasets else fp)
        self.resize = resize
        self.scale = scale
        self.style = style
        self.augment = t.augment(augment) if augment else None

    def __getitem__(self, index):
        img = fetch_image(self.paths[index])
        resized, scaled = t.get_sizes(*self.resize, self.scale)

        pipeline = t.Pipeline()
        if self.augment:
            pipeline.add(self.augment)
        if self.style:
            pipeline.add(t.style_filter())

        base = pipeline(img)
        x, y = t.resize(scaled)(base), t.resize(resized)(base)
        return x, y

    def __len__(self):
        return len(self.paths)
