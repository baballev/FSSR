import os

from torch.utils.data import Dataset
import datasets.transform as t
from .datasets import datasets
from utils import list_images, fetch_image

class BasicDataset(Dataset):
    """Single image dataset."""
    style_params = {'b': 0.2, 'c': 0.2, 's': 0.2, 'h': 0.1}

    def __init__(self, fp, scale, size, augment=False, style=False):
        self.fp = fp
        self.paths = list_images(datasets[fp] if fp in datasets else fp)
        self.sizes = t.get_sizes(size, scale)
        self.scale = scale
        self.style = style
        self.augment_name = augment
        self.augment = t.augment(augment) if augment else None

    def __getitem__(self, index):
        img = fetch_image(self.paths[index])
        resized, scaled = self.sizes

        p = t.Pipeline()
        if self.augment:
            p.add(self.augment)
        if self.style:
            p.add(t.style_filter(**self.style_params))
        base = p(img)

        x, y = t.resize(scaled)(base), t.resize(resized)(base)
        return x, y

    def __len__(self):
        return len(self.paths)

    def __str__(self):
        path = os.path.normpath(self.fp)
        name = path.split(os.sep)[-1]

        options = []
        if self.augment_name == 'augmentor':
            options.append('AGMTR')
        elif self.augment_name == 'torchvision':
            options.append('TORCHVIS')
        if self.style:
            options.append('STYLE')

        if options:
            name += '#' + '_'.join(options)
        return name

    def __repr__(self):
        """Dataset reprensentation trace."""
        string = str(self)
        string += ' ' + '-'.join(['(%ix%i)' % (h, w) for h, w in self.sizes])
        string += ' augmentation<%s>' % self.augment_name if self.augment_name else 'OFF'
        string += ' style<%s>' % self.style_params if self.style else 'OFF'
        return string

    @classmethod
    def preset(cls, name, scale, size, **kwargs):
        """BasicDataset presets"""
        if 'TORCHVISION' in name:
            kwargs['augment'] = 'torchvision'
        if 'AUGMENTOR' in name:
            kwargs['augment'] = 'augmentor'
        if 'STYLE' in name:
            kwargs['style'] = True
        fp = name.split('#')[0]
        print(kwargs)
        return cls(fp, scale, size, **kwargs)
