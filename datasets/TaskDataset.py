import os, random

from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils import list_images
from .datasets import datasets

def get_sizes(height, width, scale):
    """
    Likely to be shared with BasicDataset.

    Parameters
    ---
        width: width of the image
        height: height of the image
        limit: limits the total number of pixels in the input image
        resize: tuple(w,h) forces to resize the image to those values
    """
    resized_height, resized_width = height, width
    resized_height -= resized_height % scale
    resized_width -= resized_width % scale
    return (resized_height, resized_width), (resized_height // scale, resized_width // scale)


class ColorJitter(T.ColorJitter):
    """Wrapper on torchvision.transforms.ColorJitter to set the transform params on init."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # beware torchvision=0.9's get_params will probably return the params and not a compose
        self.transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

    def __call__(self, img):
        return self.transform(img)


class RandomGrayscale(T.RandomGrayscale):
    """Wrapper on torchvision.transforms.RandomGrayscale to set the probability on init."""
    def __init__(self, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.grayscaled = torch.rand(1) < self.p

    def __call__(self, img):
        if self.grayscaled:
            img = TF.rgb_to_grayscale(img, num_output_channels=self.num_channels)
        return img


class TaskDataset(torch.utils.data.Dataset):
    """Style-based task segmentation of the dataset."""

    def __init__(self, fp, shots, scale, augment=True, resize=None):
        self.paths = list_images(datasets[fp] if fp in datasets else fp)
        self.resize = resize
        self.scale = scale
        self.shots = shots
        self.augment = augment

    def fetch_image(self, index):
        return Image.open(self.paths[index]).convert('RGB')

    def __getitem__(self, index):
        imgs_indices = random.sample(range(len(self)), self.shots + 1)
        imgs = [self.fetch_image(i) for i in imgs_indices]
        resized, scaled = get_sizes(*self.resize, self.scale)

        augment = [
            T.RandomPerspective(),
            T.RandomRotation((-15, 15)),
            T.RandomResizedCrop(resized, scale=(0.5, 0.9), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(0.3),
            T.RandomVerticalFlip(0.3),
        ] if self.augment else []

        transform = [
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.1, 0.1)),
            RandomGrayscale(p=0.02, num_channels=3),
        ]

        resize = [T.Resize(resized, interpolation=Image.BICUBIC), T.ToTensor()]
        scale = T.Resize(scaled, interpolation=Image.BICUBIC)

        Ty = T.Compose(augment + transform + resize)
        y = torch.stack([Ty(img) for img in imgs])
        x = scale(y)

        y_spt, y_qry = y[:-1], y[-1:].squeeze(0)
        x_spt, x_qry = x[:-1], x[-1:].squeeze(0)
        return x_spt, y_spt, x_qry, y_qry

    def __len__(self):
        return len(self.paths)

