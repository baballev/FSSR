import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from utils import is_file_not_corrupted

# read https://pytorch.org/docs/stable/torchvision/transforms.html
class SBDataset(torch.utils.data.Dataset):
    """
    Style-based task partitioning of the dataset
    """
    def __init__(self, images_directory, num_shot, transform, is_valid_file=is_file_not_corrupted,
                 scale_factor=2, memory_fit_factor=4, mode='train'):
        self.is_valid_file = is_valid_file
        self.image_paths = [os.path.join(images_directory, f) for f in os.listdir(images_directory) if
                            self.is_valid_file(os.path.join(images_directory, f))]
        self.length = len(self.image_paths)
        self.transform = transform
        self.scale_factor = scale_factor
        self.num_shot = num_shot
        self.memfact = memory_fit_factor
        self.mode = mode

    def __getitem__(self, index):
        original = Image.open(self.image_paths[index]).convert('RGB')
        width, height = original.width, original.height

        if self.mode == 'train':
            resize_height = height // self.memfact
            resize_width = width // self.memfact
            while resize_height * resize_width > 393 * 510:  # Spaghetis to avoid too big tensors so it fits into 1 GPU.
                resize_height -= self.memfact
                resize_width -= int(self.memfact * (width / height))
        else:
            resize_height = height
            resize_width = width

        if resize_height % self.scale_factor != 0:
            resize_height -= (resize_height % self.scale_factor)
        if resize_width % self.scale_factor != 0:
            resize_width -= (resize_width % self.scale_factor)

        # query_label = original -> Resize -> `transform`
        query_label = self.transform(
            transforms.Resize((resize_height, resize_width),
            interpolation=Image.BICUBIC
        )(original))

        # query_data = original -> Resize
        query_data = self.transform(
            transforms.Resize((resize_height // self.scale_factor, resize_width // self.scale_factor),
            interpolation=Image.BICUBIC
        )(original))

        # transformation pipeline: <pipeline>
        augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.1, 0.1)),
            transforms.RandomPerspective(),
            transforms.RandomRotation((-15, 15)),
            transforms.RandomResizedCrop((resize_height, resize_width), scale=(0.5, 0.9), interpolation=Image.BICUBIC),
            transforms.RandomGrayscale(p=0.02),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomVerticalFlip(0.3)
        ])

        # support_label_i = original -> Resize -> `transform`
        # support_data_i = original -> <pipeline> -> Resize -> `transform`
        support_label, support_data = [], []
        for i in range(self.num_shot):
            transformed_img = augmentation(original)
            support_label.append(self.transform(transformed_img)) #! why is support_label not resized to [resize_height, resize_width]
            support_data.append(self.transform(
                transforms.Resize((resize_height // self.scale_factor, resize_width // self.scale_factor),
                                  interpolation=Image.BICUBIC)(transformed_img)))

        del original

        if self.mode == 'train':
            return torch.stack(support_data), torch.stack(support_label), query_data, query_label
        elif self.mode == 'up':
            return torch.stack(support_data), torch.stack(support_label), query_label
        else:
            raise NotImplementedError

    def __len__(self):
        return self.length

