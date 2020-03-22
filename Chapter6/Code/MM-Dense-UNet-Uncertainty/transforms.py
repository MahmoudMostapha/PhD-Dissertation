import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import avg_pool3d
from pytu.advanced.layers import onehotND


class Transforms():
    def __init__(self, callables):
        self.transforms = callables

    def __call__(self, x, y):
        data_x = x
        data_y = y
        for transform in self.transforms:
            data_x, data_y = transform(data_x, data_y)
        return data_x, data_y

def pool_inputs(data_x, data_y):
    data_x = avg_pool3d(Variable(torch.from_numpy(data_x)), 2, stride=2).data.numpy()
    return data_x, data_y

def pool_labels(data_x, data_y):
    data_y = avg_pool3d(Variable(torch.from_numpy(data_y)), 2, stride=2).data.numpy()
    return data_x, data_y

def min_max_normalize(data_x, data_y):
    mini = np.min(data_x)
    maxi = np.percentile(data_x, 99.5).astype(float)
    data_x[data_x>maxi] = maxi
    data_x = (data_x - mini) / (maxi - mini)
    return data_x, data_y

def normalize(data_x, data_y):
    data_x = data_x / 255
    return data_x, data_y

def lab_one_hot(data):
    ins, [lab_s, lab_c, lab_r] = data
    lab_s = Variable(onehotND(lab_s.data, 8))
    lab_c = Variable(onehotND(lab_c.data, 2))
    return ins, [lab_s, lab_c, lab_r]
