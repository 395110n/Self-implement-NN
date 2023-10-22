import torch
from torch import nn
from d2l import torch as d2l
import pyttsx3

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

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.1, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
engine = pyttsx3.init()
engine.say("VGG训练已完成")
engine.runAndWait()
d2l.plt.show()