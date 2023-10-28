import torch
from torch import nn

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
b1 = nn.Parameter(torch.randn(num_hiddens, requires_grad=True))

W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.randn(num_outputs, requires_grad=True))

parameters = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    # 生成大小和X一样但是元素全为0的tensor
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)
    # torch.matmul 等价于 @

loss = nn.CrossEntropyLoss()
num_epochs, lr = 20, 0.1
updater = torch.optim.SGD(parameters, lr)

