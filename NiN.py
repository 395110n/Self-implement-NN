import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, 1), nn.ReLU(), 
                         nn.Conv2d(out_channels, out_channels, 1), nn.ReLU())

net = nn.Sequential(nin_block(1, 96, kernel_size=11, strides=4, padding=0),
                    nn.MaxPool2d(kernel_size=3, stride=2), 
                    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2), nn.Dropout(p=0.5),
                    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
                    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
