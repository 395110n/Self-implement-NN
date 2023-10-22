import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
#自定义块
class MLP(nn.Module):
    def __init__(self)-> None:
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

#顺序快
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
    
#参数管理
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

#访问参数
#print(net[2].state_dict())
#OrderedDict([('weight', tensor([[ 0.0786, -0.1077, -0.0191, -0.0204,  0.1405,  
# 0.2336,  0.2195, -0.1095]])), ('bias', tensor([-0.0198]))])

"""#目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad == None)

#一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)"""

###内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

"""net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])"""

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

#参数绑定
"""shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))

net(X)
print(net[2].weight.data == net[4].weight.data)
net[2].weight.data[0, 0] = 100
print(net[2].weight.data == net[4].weight.data)"""

#自定义层
class CenteredLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
#print(layer(torch.FloatTensor([range(1, 6)])))
#可以把它作为组件合并到sequential当中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

#带参数的图层
class MyLinear(nn.Module):
    def __init__(self, in_units, units) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_units, units))
        self.bias = nn.Parameter(torch.rand(units, ))
        self.relu = MyReLU()
    
    def forward(self, X):
        X = torch.mm(X, self.weight)
        X += self.bias
        X = self.relu(X)
        return X
    
class MyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        X *= 0 <= X
        return X

dense = MyLinear(5, 3)
"""print(dense.weight)
print(dense(torch.rand(3, 5)))"""

#读写文件
#加载和保存张量
"""x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')

y = torch.zeros(4)
torch.save([x, y], 'x-files')

x2, y2 = torch.load('x-files')
print((x2, y2))"""

#加载和保存模型参数
#使用MLP类举例
net = MLP() 
#将模型的参数存储为一个叫‘mlp.params’的文件
torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())