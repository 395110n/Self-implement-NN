import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import torch.utils.data as D
import time
from datetime import datetime
from matplotlib import pyplot as plt
data_dir = 'C:\\Users\\User\\data\\cifar-10\\'
#图片存放地址

def read_csv_labels(fname):
    # fname是所有图片的标签和对应的图片名字，转成字典类型输出
    with open(fname, 'r') as outfile:
        lines = outfile.readlines()[1:]
        line_dict = {element.split(',')[0].strip(): element.split(',')[1].strip() for element in lines}
    return line_dict

def copyfile(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    n = collections.Counter(labels.values()).most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

batch_size = 32
valid_ratio = 0.1

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1), ratio=(1, 1)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform= transform_train) for folder in ['train', 'train_valid']]
#torchvision.datasets.ImageFolder函数只能接受有子文件夹的文件夹。其中子文件夹名称是类型名称，子文件夹里都是这个类型的图片

valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train_valid_test", folder),
        transform=transform_test
    )
    for folder in ('valid', 'test')
]

train_iter, train_valid_iter = [D.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]
#在变成dataloader之后，所有的标签类都已经自动转换成了hotcode一维编码

valid_iter = D.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)

test_iter = D.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        self.cv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.cv3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.conv11 = None
        if strides != 1:
            self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
    
    def forward(self, X):
        Y = nn.functional.relu(self.BN1(self.cv3_1(X)))
        Y = self.BN2(self.cv3_2(Y))
        if self.conv11:
            X = self.conv11(X)
        Y += X
        return nn.functional.relu(Y)
    
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, 10))

loss = nn.CrossEntropyLoss(reduce="none")

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = net.to(torch.device("cuda:0"))
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, loss, trainer)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
    
def train_batch(net, X, y, loss, trainer):
    X = X.to(torch.device("cuda:0"))
    y = y.to(torch.device("cuda:0"))
    trainer.zero_grad()
    y_hat = net(X)
    l = loss(y_hat, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(y_hat, y)
    return train_loss_sum, train_acc_sum

num_epochs, lr, lr_period, lr_decay, wd= 20, 2e-4, 4, 0.9, 5e-4
start = datetime.now()
train(net, train_iter, valid_iter, num_epochs, lr, wd, torch.device("cuda:0"), lr_period, lr_decay)
end = datetime.now()
print(end - start)
d2l.plt.show()

