import os

from PIL import Image
import torch, torchvision
import torchvision.transforms as T

from utils import list_images
from .datasets import datasets

class BasicDataset(torch.utils.data.Dataset):
    """
    Single image dataset fetcher
    """

    def __init__(self, dir, scale_factor=2, memory_fit_factor=4, training=True, resize=None):
        if (dir in datasets):
            dir = datasets[dir]
        self.image_paths = list_images(dir)
        self.training = training
        self.scale_factor = scale_factor
        self.memory_factor = memory_fit_factor
        self.resize = resize

    def __getitem__(self, index):
        """
            index -> image -> resize -> scale -> ?augment -> (input, label)
        """
        original = Image.open(self.image_paths[index]).convert('RGB')
        width, height = original.width, original.height

        if self.resize is not None: # force resize with arbitrary values
            resize_height, resize_width = self.resize

        elif self.training:
            resize_height = height // self.memory_factor
            resize_width = width // self.memory_factor

            while resize_height * resize_width > 393 * 510:  # avoid too big tensors for GPU
                resize_height -= self.memory_factor
                resize_width -= int(self.memory_factor * (width / height))

        else:
            resize_height = height
            resize_width = width

        resize_height -= resize_height % self.scale_factor
        resize_width -= resize_width % self.scale_factor
        resized = (resize_height, resize_width)

        augmentation = T.Compose([
            T.RandomPerspective(),
            T.RandomRotation((-15, 15)),
            T.RandomResizedCrop(resized, scale=(0.5, 0.9), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(0.3),
            T.RandomVerticalFlip(0.3)
        ])

        scaled = (resized[0] // self.scale_factor, resized[1] // self.scale_factor)

        # print('original:', (height, width))
        # print('resized:', resized)
        # print('scaled:', scaled)

        resize = [T.Resize(resized, interpolation=Image.BICUBIC)]
        resize_and_scale = [T.Resize(scaled, interpolation=Image.BICUBIC)]
        to_tensor = [T.ToTensor()]

        image = augmentation(original) if self.training else original
        return T.Compose(resize_and_scale + to_tensor)(image), T.Compose(resize + to_tensor)(image)

    def __len__(self):
        return len(self.image_paths)

    def get_image_name(self, index):
        return self.image_paths[index]
