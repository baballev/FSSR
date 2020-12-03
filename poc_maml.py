import os, warnings
from copy import deepcopy

import torch
import torch.optim
import torch.nn as nn
from torch.nn.functional import l1_loss

from model import MAML

warnings.filterwarnings('ignore', message='CUDA init')

# for reproducibility
if os.path.isfile('test.pt'):
    X, y, model = torch.load('test.pt')
else:
    X = torch.rand(10, 20)
    y = torch.rand(10, 10)
    model = nn.Linear(20, 10)
    torch.save((X, y, model), 'test.pt')


def vanilla_train(X, y, model):
    ## Vanilla training
    X, y, model = deepcopy(X), deepcopy(y), deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    error = l1_loss(model(X), y)
    print('%f (initial error)' % error)

    error.backward()

    optimizer.step()
    error = l1_loss(model(X), y)
    print('%f (after one update)' % error)


def meta_train(X, y, model):
    ## Meta training
    X, y, model = deepcopy(X), deepcopy(y), deepcopy(model)
    maml = MAML(model, lr=0.01)
    optimizer = torch.optim.Adam(maml.parameters(), lr=0.001)
    cloned = maml.clone()

    error = l1_loss(cloned(X), y)
    print('%f (initial error)' % error)

    cloned.adapt(error)
    error = l1_loss(cloned(X), y)
    print('%f (after adapting cloned model)' % error)

    error.backward()

    error = l1_loss(maml(X), y)
    print('%f (before backward pass - didn\'t changed because adaption was on cloned model)' % error)

    optimizer.step()

    error = l1_loss(maml(X), y)
    print('%f (after backward pass)' % error)

    cloned = maml.clone()
    error = l1_loss(cloned(X), y)
    cloned.adapt(error)
    error = l1_loss(cloned(X), y)
    print('%f (after finetuning)' % error)


vanilla_train(X, y, model)
print('*'*100)
meta_train(X, y, model)
