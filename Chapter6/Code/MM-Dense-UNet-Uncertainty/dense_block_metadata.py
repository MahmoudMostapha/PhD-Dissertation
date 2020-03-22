import torch.nn as nn
import torch
from mixed_block import get_mixed_block


class _MixedDenseBlock(nn.Module):
    def __init__(self, in_conv, in_meta, kernel_size, *args, n_layer=1, growth_rate=1, **kwargs):
        super().__init__()
        self.in_conv = in_conv
        self.in_meta = in_meta
        self.n_layer = n_layer
        self.growth_rate = growth_rate

        self.out_conv = self.in_conv + self.growth_rate * self.n_layer
        self.out_meta = self.in_meta + self.growth_rate * self.n_layer
        self.layers = nn.ModuleList()

        if self._alternate:
            kernels = [(kernel_size, kernel_size, 1), (1, 1, kernel_size)]
        else:
            kernels = [kernel_size] * n_layer

        for i in range(n_layer):
            kernel_size = kernels[i % len(kernels)]
            layer = self._layertype(
                    *args,
                    in_conv=in_conv,
                    out_conv=growth_rate,
                    in_meta=in_meta,
                    out_meta=growth_rate,
                    kernel_size=kernel_size,
                    **kwargs)
            in_conv += growth_rate
            in_meta += growth_rate
            self.layers.append(layer)

    def forward(self, img, meta):
        stack_img = img
        stack_meta = meta
        for i, layer in enumerate(self.layers):
            new_img, new_meta = layer(stack_img, stack_meta)
            stack_img = torch.cat((stack_img, new_img), 1)
            stack_meta = torch.cat((stack_meta, new_meta), 1)
        return stack_img, stack_meta

class MixedDenseBlock1D(_MixedDenseBlock):
    def __init__(self, *args, **kwargs):
        self._alternate = False
        self._layertype = get_mixed_block(1)
        super().__init__(*args, **kwargs)


class MixedDenseBlock2D(_MixedDenseBlock):
    def __init__(self, *args, **kwargs):
        self._alternate = False
        self._layertype = get_mixed_block(2)
        super().__init__(*args, **kwargs)

class MixedDenseBlockFake3D(_MixedDenseBlock):
    def __init__(self, *args, **kwargs):
        self._alternate = True
        self._layertype = get_mixed_block(3)
        super().__init__(*args, **kwargs)

class MixedDenseBlock3D(_MixedDenseBlock):
    def __init__(self, *args, **kwargs):
        self._alternate = False
        self._layertype = get_mixed_block(3)
        super().__init__(*args, **kwargs)


_mixed_dense_blocks = {
    1: MixedDenseBlock1D,
    2: MixedDenseBlock2D,
    2.5: MixedDenseBlockFake3D,
    3: MixedDenseBlock3D
}


def get_mixed_dense_block(dim):
    """N-dimension getter for Mixed Dense Blocks.

    :param dim: Dimension of the Dense Block wanted.
    :type dim: int
    :return: A Dense Block Constructor in the right dimension.

    """

    if dim not in _mixed_dense_blocks:
        raise Exception('Dimensions should be either 1, 2 or 3, not {}'.format(dim))
    return _mixed_dense_blocks[dim]
