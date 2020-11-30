import torch
import Augmentor
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def get_sizes(height, width, scale):
    """
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


def augment(library):
    if library == 'torchvision':
        return T.Compose([
            T.RandomPerspective(),
            T.RandomRotation((-15, 15)),
            T.RandomResizedCrop((256, 512), scale=(0.5, 0.9), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(0.3),
            T.RandomVerticalFlip(0.3)
        ])
    elif library == 'augmentor':
        p = Augmentor.Pipeline()
        p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.flip_top_bottom(probability=0.3)
        return p.torch_transform()
    else:
        raise NotImplementedError


def style_filter():
    return T.Compose([
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.1, 0.1)),
        RandomGrayscale(p=0.02, num_channels=3),
    ])

def resize(size):
    return T.Compose([
        T.Resize(size, interpolation=Image.BICUBIC),
        T.ToTensor(),
    ])

class Pipeline:
    transforms = []

    def __call__(self, x):
        return T.Compose(self.transforms)(x)

    def add(self, transform):
        self.transforms.append(transform)
