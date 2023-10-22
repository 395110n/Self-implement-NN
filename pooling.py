import torch
from torch import nn
from d2l import torch as d2l

def mypool2d(X, pool_size, mode="max"):
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == "avg":
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

#X = torch.arange(9, dtype=torch.float32).reshape(3, 3)
X = torch.arange(16, dtype=torch.float32).reshape((1,1,4,4))
pool2d = nn.MaxPool2d(3,padding=1, stride=2)
print(pool2d(X))

