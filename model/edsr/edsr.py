"""
Credit goes to the following repo for the content of this file:
https://github.com/thstkdgus35/EDSR-PyTorch
"""

import torch.nn as nn

from .common import Conv2d, MeanShift, ResBlock, Upsampler


class EDSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, scale, rgb_range=1, n_colors=3, res_scale=1):
        super(EDSR, self).__init__()

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        kernel_size = 3

        # define head module
        m_head = [Conv2d(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = []
        for _ in range(n_resblocks):
            m_body.append(ResBlock(n_feats, kernel_size, res_scale=res_scale))
        m_body.append(Conv2d(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(scale, n_feats),
            Conv2d(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
