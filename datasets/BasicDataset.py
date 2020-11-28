from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from utils import list_images, fetch_image
import datasets.transform as transform
from .datasets import datasets

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
        resized, scaled = transform.get_sizes(*self.resize, self.scale)

        pipeline = []
        if self.augment:
            pipeline.append(transform.augment())
        if self.style:
            pipeline.append(transform.style_filter())
        pipeline += [transform.resize(resized), ToTensor()]
        scale = transform.resize(scaled)

        T = Compose(pipeline)
        y = T(img)
        x = scale(y)
        return x, y

    def __len__(self):
        return len(self.paths)
