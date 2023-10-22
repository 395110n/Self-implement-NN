import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.data import Dataset, DataLoader
import random

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5

true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

train_data = d2l.synthetic_data(true_w, true_b, n_train)

train_iter = d2l.load_array(train_data, batch_size)

test_data = d2l.synthetic_data(true_w, true_b, n_test)

test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.abs())

def train(lamda):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lamda * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch+1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
            d2l.plt.draw()
            d2l.plt.pause(0.01)
    print('w的L2范数是: ', torch.norm(w).item())
train(lamda=5)
d2l.plt.show()

"""def generate_features(n_train, num_inputs):
    features = torch.zeros(n_train, num_inputs)
    for num in range(n_train):
        x = true_w * torch.randn(200).reshape(true_w.size()[0], 1)
        x = x.reshape(1, -1)
        features[num, :] += x.reshape(features[0, :].size())
    return features

def generate_labels_with_normal(true_w, true_b, n_train):
    features = generate_features(n_train, true_w.size()[0])
    true_w = true_w.reshape(1, -1)
    real_data = true_w * features
    real_data = real_data.sum(axis=1, keepdim=True)
    real_data += true_b
    train_labels = real_data + torch.normal(0, 0.01, real_data.size())
    return features, train_labels

features, labels = generate_labels_with_normal(true_w, true_b, n_train)
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
train_iter = data_iter(batch_size, features, labels)"""
