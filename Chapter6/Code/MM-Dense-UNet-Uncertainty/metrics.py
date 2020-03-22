import torch
from pytu.advanced.layers import onehotND

def get_dice(chan=0, name='dice'):

    return dice(chan, name)

class dice:

    def __init__(self, chan, name='dice'):

        self.chan = chan
        self.__name__ = name

    def __call__(self, x, y):
        x, _, _ = x
        y, _, _ = y
        output_nc = x.size()[1]
        x = torch.argmax(x, dim=1, keepdim=True)
        x = onehotND(x, output_nc)

        sums = torch.sum(x[:, self.chan]) + torch.sum(y[:, self.chan])
        if sums == 0:
            return sums
        return (torch.sum(x[:, self.chan] * y[:, self.chan]) * 2.0) / sums


def accuracy(x, y):
    _, x, _ = x
    _, y, _ = y
    if x.size() != y.size():
        x = x.view_as(y)
    _, mx = x.max(1)
    _, my = y.max(1)
    res = torch.mean((mx == my).type_as(x))
    return res

def mae(x, y):
    _, _, x = x
    _, _, y = y

    x = x * (2.7183 - 1) + 1
    x = torch.log(x) * 20.0 + 10.0
    x = torch.round(x)

    x[x < 10.0] = 10.0
    x[x > 30.0] = 30.0

    y = y * (2.7183 - 1) + 1
    y = torch.log(y) * 20.0 + 10.0
    y = torch.round(y)

    y[y < 10.0] = 10.0
    y[y > 30.0] = 30.0

    return torch.abs(x - y)