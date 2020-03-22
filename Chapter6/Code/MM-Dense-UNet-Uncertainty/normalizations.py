import torch.nn as nn
import torch

class _MomentNorm(nn.Module):
    def __init__(self, channels, high_order_mode=False, learn_weight=False, learn_bias=False, epsilon=1e-5):
        super().__init__()
        self.channels = channels
        self.high_order_mode = high_order_mode
        self.learn_weight = learn_weight
        self.learn_bias = learn_bias
        self.epsilon = epsilon

        if learn_weight:
            self.register_parameter(name='weight', data=nn.Parameter(torch.Tensor(1, channels, 1, 1)))
            torch.nn.init.constant_(self.weight, 1)
            self.apply_weight = lambda x: x*self.weight
        else:
            self.apply_weight = lambda x: x
        if self.learn_bias:
            self.register_parameter(name='bias', data=nn.Parameter(torch.Tensor(1, channels, 1, 1)))
            torch.nn.init.constant_(self.bias, 0)
            self.apply_bias = lambda x: x + self.bias
        else:
            self.apply_bias = lambda x: x

    def forward(self, input):
        mean = input.mean(dim=self._dims, keepdim=True)
        output = input - mean
        n = 1
        for dim in self._dims:
            n *= output.shape[dim]
        std = torch.sqrt((output*output).sum(dim=self._dims, keepdim=True)/float(n - 1)) + self.epsilon
        alpha = self.apply_weight(1/std)
        if not self.high_order_mode:
            return self.apply_bias(output * alpha), mean, std
        else:
            skew = (output*output*output).sum(dim=self._dims, keepdim=True)/std**2 + self.epsilon
            skew*= float(n)/(float(n-1)*float(n-2))
            kurt = (output*output*output*output).sum(dim=self._dims, keepdim=True)/std**3 + self.epsilon
            kurt*= (float(n)*float(n+1))/(float(n-1)*float(n-2)*float(n-3))
            kurt-= (3*float(n-1)**2)/(float(n-2)*float(n-3))
            return self.apply_bias(output * alpha), mean, std, skew, kurt


class MomentNorm1D(_MomentNorm):
    def __init__(self, *args, **kwargs):
        self._dims = (2)
        super().__init__(*args, **kwargs)

class MomentNorm2D(_MomentNorm):
    def __init__(self, *args, **kwargs):
        self._dims = (2, 3)
        super().__init__(*args, **kwargs)

class MomentNorm3D(_MomentNorm):
    def __init__(self, *args, **kwargs):
        self._dims = (2, 3, 4)
        super().__init__(*args, **kwargs)

_moment_norms = {
    1: MomentNorm1D,
    2: MomentNorm2D,
    3: MomentNorm3D
}

def get_moment_norm(dim):
    if dim not in _moment_norms:
        raise Exception('Mixed block: spatial dimensions should be either 1, 2 or 3, not {}'.format(dim))
    return _moment_norms[dim]

class GlobalNorm0D(nn.Module):
    def __init__(self, channels, learning_rate=0.0001, learn_weight=False, learn_bias=False, epsilon=1e-5):
        super().__init__()
        self.channels = channels
        self.learning_rate = learning_rate
        self.learn_weight = learn_weight
        self.learn_bias = learn_bias
        self.epsilon = epsilon
        self.register_buffer(name='running_mean', tensor=torch.zeros(1, channels))
        self.register_buffer(name='running_var', tensor=torch.ones(1, channels))
        if learn_weight:
            self.register_parameter(name='weight', data=nn.Parameter(torch.Tensor(1, channels)))
            nn.init.constant_(self.weight, 1)
            self.apply_weight = lambda x: x * self.weight
        else:
            self.apply_weight = lambda x: x
        if self.learn_bias:
            self.register_parameter(name='bias', data=nn.Parameter(torch.Tensor(1, channels)))
            nn.init.constant_(self.bias, 0)
            self.apply_bias = lambda x: x + self.bias
        else:
            self.apply_bias = lambda x: x

    def forward(self, input):
        if self.training:
            with torch.no_grad():
                mean = input.mean(dim=0, keepdim=True)
                self.running_mean *= 1-self.learning_rate
                self.running_mean += self.learning_rate*mean

        vec = input - self.running_mean

        if self.training:
            with torch.no_grad():
                var = (vec*vec).sum(dim=0, keepdim=True)/float(vec.shape[0])
                self.running_var *= 1-self.learning_rate
                self.running_var += self.learning_rate*var
                self.running_var += self.epsilon

        alpha = self.apply_weight(self.running_var ** (-0.5))
        return self.apply_bias(vec * alpha)


class GlobalNormND(nn.Module):
    def __init__(self, channels, learning_rate=0.0001, learn_weight=False, learn_bias=False, epsilon=1e-5):
        super().__init__()
        self.channels = channels
        self.learning_rate = learning_rate
        self.learn_weight = learn_weight
        self.learn_bias = learn_bias
        self.epsilon = epsilon
        self.register_buffer(name='running_mean', tensor=torch.zeros(1, channels))
        self.register_buffer(name='running_var', tensor=torch.ones(1, channels))
        if learn_weight:
            self.register_parameter(name='weight', data=nn.Parameter(torch.Tensor(1, channels)))
            nn.init.constant_(self.weight, 1)
            self.apply_weight = lambda x: x * self.weight
        else:
            self.apply_weight = lambda x: x
        if self.learn_bias:
            self.register_parameter(name='bias', data=nn.Parameter(torch.Tensor(1, channels)))
            nn.init.constant_(self.bias, 0)
            self.apply_bias = lambda x: x + self.bias
        else:
            self.apply_bias = lambda x: x

    def forward(self, input):
        vec = input.view(input.shape[0], input.shape[1], -1)

        if self.training:
            with torch.no_grad():
                mean = vec.mean(dim=[0, 2])
                self.running_mean *= 1-self.learning_rate
                self.running_mean += self.learning_rate*mean.unsqueeze(0)

        vec -= self.running_mean

        if self.training:
            with torch.no_grad():
                var = (vec*vec).sum(dim=[0, 2])/float(vec.shape[0]*vec.shape[2])
                self.running_var *= 1-self.learning_rate
                self.running_var += self.learning_rate*var.unsqueeze(0)
                self.running_var += self.epsilon

        alpha = self.apply_weight(self.running_var ** (-0.5))
        return self.apply_bias(vec * alpha)
