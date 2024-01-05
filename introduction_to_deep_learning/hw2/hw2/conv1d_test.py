import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import Conv1D

## initialize your layer and PyTorch layer
net1 = Conv1D(8, 12, 3, 2)
net2 = torch.nn.Conv1d(8, 12, 3, 2)
## initialize the inputs
x1 = np.random.rand(3, 8, 20)
x2 = Variable(torch.tensor(x1), requires_grad=True)
## Copy the parameters from the Conv1D class to PyTorch layer
net2.weight = nn.Parameter(torch.tensor(net1.W))
net2.bias = nn.Parameter(torch.tensor(net1.b))
## Your forward and backward
y1 = net1(x1)
b, c, w = y1.shape
delta = np.random.randn(b, c, w)
dx = net1.backward(delta)
## PyTorch forward and backward
y2 = net2(x2)
delta = torch.tensor(delta)
y2.backward(delta)


## Compare
def compare(x, y, var=""):
    y = y.detach().numpy()
    print("Diff of {}".format(var), abs(x - y).max())
    return


compare(y1, y2, 'y')
compare(dx, x2.grad, 'dx')
compare(net1.dW, net2.weight.grad, 'dw')
compare(net1.db, net2.bias.grad, 'db')
