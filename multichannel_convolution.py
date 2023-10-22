import torch
from d2l import torch as d2l

def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.concat((torch.arange(9).reshape(3, 3), torch.arange(1, 10).reshape(3, 3))).reshape(2, 3, 3)
K = torch.concat((torch.arange(4).reshape(2, 2), torch.arange(1, 5).reshape(2, 2))).reshape(2, 2, 2)

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

#K = torch.stack((K, K+1, K+2), 0)
#把三个矩阵按照第0维叠起来

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print(Y1 == Y2)