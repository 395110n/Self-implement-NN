import torch
from torch import nn
from d2l import torch as d2l

dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), \
                    nn.Dropout(dropout1), nn.Linear(256, 256),\
                    nn.ReLU(), nn.Dropout(dropout2), nn.Linear(256, 10))

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal(m.weight, std=0.01)

net.apply(init_weight)