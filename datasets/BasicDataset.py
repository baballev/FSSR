import os
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from utils import list_images

class BasicDataset(torch.utils.data.Dataset):
  """
  Single image dataset fetcher
  """
  def __init__(self, images_directory, scale_factor=2, memory_fit_factor=4, training=True):
    self.image_paths = list_images(images_directory)
    self.training = training
    self.scale_factor = scale_factor
    self.memory_factor = memory_fit_factor

  def __getitem__(self, index):
    original = Image.open(self.image_paths[index]).convert('RGB')
    width, height = original.width, original.height

    resize_height = height // self.memory_factor
    resize_width = width // self.memory_factor
    while resize_height * resize_width > 393 * 510:  # avoid too big tensors
      resize_height -= self.memory_factor
      resize_width -= int(self.memory_factor * (width / height))

    resize_height -= resize_height % self.scale_factor
    resize_width -= resize_width % self.scale_factor

    resize_dimensions = (resize_height, resize_width)
    resize_factors = (resize_height // self.scale_factor, resize_width // self.scale_factor)

    augmentation = T.Compose([
      T.RandomPerspective(),
      T.RandomRotation((-15, 15)),
      T.RandomResizedCrop(resize_dimensions, scale=(0.5, 0.9), interpolation=Image.BICUBIC),
      T.RandomHorizontalFlip(0.3),
      T.RandomVerticalFlip(0.3)
    ])

    resize = [T.Resize(resize_factors, interpolation=Image.BICUBIC)]
    to_tensor = [T.ToTensor()]

    image = augmentation(original) if self.training else original
    return T.Compose(resize + to_tensor)(image), T.Compose(to_tensor)(image)

  def __len__(self):
    return len(self.image_paths)
