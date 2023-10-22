import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import torch.utils.data as data
import matplotlib.pyplot as plt
train_imgs = torchvision.datasets.ImageFolder("C:\\Users\\User\\data\\hotdog\\test")
test_imgs = torchvision.datasets.ImageFolder("C:\\Users\\User\\data\\hotdog\\train")

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(in_features=finetune_net.fc.in_features, out_features=2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

def train_fine_tuning(net, learning_rate, batch_size = 128, num_epochs=5, param_group=True):
    train_iter = data.DataLoader(
        torchvision.datasets.ImageFolder("C:\\Users\\User\\data\\hotdog\\train",
                                          transform=train_augs),
        batch_size=batch_size, shuffle=True
    )
    test_iter = data.DataLoader(
        torchvision.datasets.ImageFolder("C:\\Users\\User\\data\\hotdog\\test",
                                         transform=test_augs)
                                        ,batch_size=batch_size
    )
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ['fc.weight', 'fc.bias']]
        trainer = torch.optim.SGD([{'params': params_1x}, 
                                   {'params' : net.fc.parameters(),
                                   'lr' : learning_rate * 10}], 
                                   lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

"""
加了微调
train_fine_tuning(finetune_net, 5e-5)
plt.show()"""

#不加微调
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
plt.show()