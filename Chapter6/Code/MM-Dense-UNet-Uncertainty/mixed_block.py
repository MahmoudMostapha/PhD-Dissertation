import torch.nn as nn
import torch
from pytu.advanced.layers import get_padding
from normalizations import get_moment_norm, GlobalNorm0D
from dirac_orthogonal import dirac_orthogonal_
from pool import get_pool

class _MixedBlock(nn.Module):
    def __init__(self,
                 convtype,
                 padtype,
                 dim,
                 in_conv,
                 out_conv,
                 kernel_size,
                 in_meta,
                 out_meta,
                 *args,
                 use_global_pooling=False,
                 stride=1,
                 activation=nn.functional.relu,
                 padding_type='valid',
                 padding_value=0,
                 init_behaviour='normalized',
                 high_order_mode=False,
                 **kwargs
    ):
        super().__init__()
        self._convtype = convtype
        self._padtype = padtype
        self.dim = dim
        self.out_meta = out_meta
        self.high_order_mode = high_order_mode
        self.use_global_pooling = use_global_pooling

        no_stat = 4 if high_order_mode else 2

        pad = 0
        self.padding = None
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * self.dim
        if padding_type == 'same' and all([k % 2 == 1 for k in kernel_size]):
            pad = [k // 2 for k in kernel_size]
        elif padding_type == 'zeros':
            pad = padding_value
        elif padding_type in ['same', 'same_replicate', 'replicate']:
            self.padding = self._padtype(kernel_size, stride, padding_type, pad=padding_value)
        elif padding_type != 'valid':
            raise Exception('Padding type should be either zeros, same, same_replicate, replicate or valid.')

        self.img_norm = self._moment_norm(channels=in_conv, high_order_mode=high_order_mode)
        self.meta_norm = GlobalNorm0D(channels=in_conv*no_stat+in_meta)

        self.denorm = nn.Linear(in_conv*no_stat+in_meta, in_conv)
        if init_behaviour == 'normalized':
            torch.nn.init.constant_(self.denorm.weight, 0)
            torch.nn.init.constant_(self.denorm.bias, 1)
        elif init_behaviour == 'denormalized':
            torch.nn.init.eye_(self.denorm.weight)
            torch.nn.init.constant_(self.denorm.bias, 0)

        self.activation = activation
        if self.activation not in [None, 'None'] and self.activation.__name__ not in ['softmax']:
            gain = torch.nn.init.calculate_gain(self.activation.__name__)
        else:
            gain = 1

        if out_meta > 0:
            self.meta2meta = nn.Linear(in_conv*no_stat+in_meta, out_meta, bias=True)
            torch.nn.init.orthogonal_(self.meta2meta.weight, gain * 1e-2)
            #torch.nn.init.orthogonal_(self.meta2meta.weight, 0)
        else:
            self.meta2meta = lambda x: torch.Tensor([])

        if out_conv > 0:
            if self.use_global_pooling:
                self.meta2img = nn.Linear(in_conv * no_stat + in_meta, in_conv, bias=False)
                self.conv = self._convtype(in_conv,
                                           in_conv,
                                           kernel_size,
                                           padding=pad,
                                           stride=stride,
                                           bias=True)
                self.fc = nn.Linear(in_conv, out_conv)
            else:
                self.meta2img = nn.Linear(in_conv * no_stat + in_meta, out_conv, bias=False)
                self.conv = self._convtype(in_conv,
                                           out_conv,
                                           kernel_size,
                                           padding=pad,
                                           stride=stride,
                                           bias=True)
            torch.nn.init.orthogonal_(self.meta2img.weight, gain * (2 ** -0.5) * 1e-2)
            #torch.nn.init.orthogonal_(self.meta2img.weight, 0.)
            dirac_orthogonal_(self.conv.weight, gain * (2 ** -0.5) * 1e-2)
            torch.nn.init.constant_(self.conv.bias, 0)
        else:
            self.meta2img = lambda x: torch.Tensor([])
            self.conv = lambda x: torch.Tensor([])

    def forward(self, img, meta):

        if not self.high_order_mode:
            img, mean, std = self.img_norm(img)
            meta = self.meta_norm(torch.cat([self._squeeze(std), self._squeeze(mean), meta], dim=1))
        else:
            img, mean, std, skew, kurt = self.img_norm(img)
            meta = self.meta_norm(torch.cat([self._squeeze(skew), self._squeeze(kurt), self._squeeze(std), self._squeeze(mean), meta], dim=1))

        img *= self._unsqueeze(self.denorm(meta))

        if self.padding is not None:
            img = self.padding(img)

        img = self.conv(img) + self._unsqueeze(self.meta2img(meta))
        meta = self.meta2meta(meta)

        if self.use_global_pooling:
            img = self.global_pooling()(img)
            img = self._squeeze(img)
            img = self.fc(img)

        if self.activation is not None:
            if self.activation.__name__ == 'softmax':
                img = self.activation(img, dim=1)
            else:
                img  = self.activation(img)
                meta = self.activation(meta)


        return (img, meta) if self.out_meta > 0 else img

class MixedBlock1D(_MixedBlock):
    def __init__(self, *args, **kwargs):
        self._unsqueeze = lambda x: x.unsqueeze(2)
        self._squeeze = lambda x: x.squeeze(2)
        self._moment_norm = get_moment_norm(1)
        self.global_pooling = get_pool(1, 'global_max')
        _MixedBlock.__init__(self, nn.Conv1d, get_padding(1), 1, *args, **kwargs)


class MixedBlock2D(_MixedBlock):
    def __init__(self, *args, **kwargs):
        self._unsqueeze = lambda x: x.unsqueeze(2).unsqueeze(3)
        self._squeeze = lambda x: x.squeeze(3).squeeze(2)
        self._moment_norm = get_moment_norm(2)
        self.global_pooling = get_pool(2, 'global_max')
        _MixedBlock.__init__(self, nn.Conv2d, get_padding(2), 2, *args, **kwargs)


class MixedBlock3D(_MixedBlock):
    def __init__(self, *args, **kwargs):
        self._unsqueeze = lambda x: x.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        self._squeeze = lambda x: x.squeeze(4).squeeze(3).squeeze(2)
        self._moment_norm = get_moment_norm(3)
        self.global_pooling = get_pool(3, 'global_max')
        _MixedBlock.__init__(self, nn.Conv3d, get_padding(3), 3, *args, **kwargs)

_mixed_blocks = {
    1: MixedBlock1D,
    2: MixedBlock2D,
    3: MixedBlock3D
}

def get_mixed_block(dim):
    if dim not in _mixed_blocks:
        raise Exception('Mixed block: spatial dimensions should be either 1, 2 or 3, not {}'.format(dim))
    return _mixed_blocks[dim]
