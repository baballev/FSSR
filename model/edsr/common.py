"""
Credit goes to the following repo for the content of this file:
https://github.com/thstkdgus35/EDSR-PyTorch
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, sign=-1, rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0)):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, res_scale, bias=True):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            Conv2d(n_feats, n_feats, kernel_size, bias=bias),
            nn.ReLU(True),
            Conv2d(n_feats, n_feats, kernel_size, bias=bias))

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(Conv2d(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(Conv2d(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
