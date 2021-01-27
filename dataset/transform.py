import torch
import Augmentor
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def get_sizes(size, scale):
    """
    Parameters
    ---
        size: size of the input image (height, width)
        scale: scale factor applied to HR image
    """
    h, w = size
    return (h, w), (h*scale, w*scale)


class ColorJitter(T.ColorJitter):
    """Wrapper on torchvision.transforms.ColorJitter to set the transform params on init."""
    def __init__(self, **kwargs):
        super(ColorJitter, self).__init__(**kwargs)
        # beware torchvision=0.9's get_params will probably return the params and not a compose
        self.transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

    def __call__(self, img):
        return self.transform(img)


class RandomGrayscale(T.RandomGrayscale):
    """Wrapper on torchvision.transforms.RandomGrayscale to set the probability on init."""
    def __init__(self, num_channels, **kwargs):
        super(RandomGrayscale, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.grayscaled = torch.rand(1) < self.p

    def __call__(self, img):
        if self.grayscaled:
            img = TF.rgb_to_grayscale(img, num_output_channels=self.num_channels)
        return img


def augment(flag, size):
    p = Augmentor.Pipeline()
    if flag:
        p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.2, percentage_area=0.8)
        p.flip_top_bottom(probability=0.3)
        p.crop_by_size(probability=1, width=size[1], height=size[0], centre=False)
        # p.resize(probability=1, width=size[1], height=size[0])
    else:
        p.crop_by_size(probability=1, width=size[1], height=size[0], centre=True)
        #p.resize(probability=1, width=size[1], height=size[0])
    return p.torch_transform()


def style_filter(b, c, s, h):
    #return RandomGrayscale(p=1, num_channels=3)
    return ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)

def resize(size):
    return T.Compose([
        T.Resize(size, interpolation=Image.BICUBIC),
        T.ToTensor(),
    ])

def tensor(x):
    return T.ToTensor()(x)

class Pipeline:
    def __init__(self, *transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        return T.Compose(self.transforms)(x)

    def add(self, transform):
        self.transforms.append(transform)

    def __len__(self):
        return len(self.transforms)
