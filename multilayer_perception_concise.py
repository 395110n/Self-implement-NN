import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(),nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
batch_size, lr, num_epochs = 256, 0.1, 10

loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(), lr = lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)