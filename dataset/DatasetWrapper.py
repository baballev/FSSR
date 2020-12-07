import os

from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
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
        string = str(self) \
               + ' ' + '-'.join(['(%ix%i)' % (h, w) for h, w in self.sizes]) \
               + ' augmentation<%s>' % (self.augment_name if self.augment_name else 'OFF') \
               + ' style<%s>' % (self.style_params if self.style else 'OFF')
        return string

    @classmethod
    def preset(cls, name, **kwargs):
        """BasicDataset presets"""
        if 'TORCHVISION' in name:
            kwargs['augment'] = 'torchvision'
        if 'AUGMENTOR' in name:
            kwargs['augment'] = 'augmentor'
        if 'STYLE' in name:
            kwargs['style'] = True
        fp = name.split('#')[0]
        return cls(fp, **kwargs)
