import dataset.transform as t
from .datasets import datasets
from .DatasetWrapper import DatasetWrapper
from utils import list_images, fetch_image

class BasicDataset(DatasetWrapper):
    """Single image dataset."""
    style_params = {'b': 0.4, 'c': 0.4, 's': 0.4, 'h': 0.2}

    def __init__(self, fp, scale, size, augment=False, style=False):
        self.fp = fp
        self.paths = list_images(datasets[fp] if fp in datasets else fp)
        self.sizes = t.get_sizes(size, scale)
        self.style = style
        self.augment_name = augment
        self.augment = t.augment(augment, self.sizes[0])
        self.scale = t.resize(self.sizes[1])

    def __getitem__(self, index):
        img = fetch_image(self.paths[index])

        p = t.Pipeline(self.augment)
        if self.style:
            p.add(t.style_filter(**self.style_params))
        base = p(img)

        x, y = self.scale(base), t.tensor(base)
        return x, y

    def __len__(self):
        return len(self.paths)
