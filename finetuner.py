import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from copy import deepcopy
import os
from math import log2, floor

os.chdir(os.path.dirname(os.path.realpath(__file__)))
from models import ResBlock, MeanShift, PoolBlock, ConvBlock



class FineTuner(nn.Module):
    def __init__(self, network, fine_tuning_depth):
        super.__init__(self, FineTuner)
        self.config = network.getconfig()
        self.network = network
        self.depth = fine_tuning_depth
        target_config = self.config[-fine_tuning_depth:]
        parameters_size = 0
        for type, conf in target_config:
            if type == 'conv2d':
                parameters_size += conf[0]*conf[1]*conf[2]*conf[3] + conf[0]
            elif type == 'convt2d':
                parameters_size += conf[0]*conf[1]*conf[2]*conf[3] + conf[1]
            elif type == 'linear':
                parameters_size += conf[0]*conf[1]
            elif type == 'bn': # No Idea wether a meta learner could be able to figure out how to give the right parameters for a batch normalization
                raise NotImplementedError
            elif type == 'prelu':
                raise NotImplementedError # ToDo
            elif type == 'reblock_leakyrelu' or type == 'resblock_relu':
                parameters_size += (conf[0]*conf[1]*conf[2]*conf[3] + conf[0])*2
            else:
                pass
        print("Setuping meta fine-tuner network for " + str(parameters_size) + " parameters.", flush=True)
        self.block_number = int(floor(log2(parameters_size) - 3)) # -3 because we start with RGB(3) to 8 channels
        print("Making an architecture of depth " + str(self.block_number + 1) + ".", flush=True)

        m = []
        m.append(nn.Conv2d(3, 8, kernel_size=3, padding=0))
        for i in range(self.block_number):
            if i%2 == 1:
                m.append(PoolBlock(2**(3+i), 2**(3+i+1), act=nn.LeakyReLU(inplace=True)))
            else:
                m.append(ConvBlock(2**(3+i), 2**(3+i+1), act=nn.LeakyReLU(inplace=True)))
        m.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.body = nn.Sequential(*m)
        self.tail = nn.Sequential(nn.Linear(self.block_number, parameters_size))

    def forward(self, x_spt, x_qry):
        x_spt = self.body(x_spt)
        x_spt = x_spt.view(-1, 2**(3 + self.block_number))
        x_spt = self.tail(x_spt) # x_spt is now a vector representing the parameters that needs to be loaded into the SR network.
        # ToDo: Load the parameters using the network config and code the forward pass


        return x

