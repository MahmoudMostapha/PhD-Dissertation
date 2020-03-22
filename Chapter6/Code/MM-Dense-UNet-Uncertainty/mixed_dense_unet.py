from dense_block_metadata import get_mixed_dense_block
from mixed_block import get_mixed_block
import torch.nn as nn
import torch
from pytu.misc import Interpolate
from pytu.advanced.layers.pool import get_pool

class _MixedDenseUNet(nn.Module):
    def __init__(self,
                 *args,
                 in_conv=1,
                 out_conv_seg=4,
                 out_conv_class=3,
                 out_conv_reg=1,
                 in_meta=1,
                 out_meta=0,
                 growth_rate=12,
                 n_layer=1,
                 n_pool=3,
                 activation=nn.functional.relu,
                 kernel_size=3,
                 final_kernel_size=3,
                 **kwargs):

        super().__init__()
        self.in_conv = in_conv
        self.out_conv_seg = out_conv_seg
        self.out_conv_class = out_conv_class
        self.out_conv_reg = out_conv_reg
        self.in_meta = in_meta
        self.out_meta = out_meta
        self.growth_rate = growth_rate
        self.n_layer = n_layer
        self.n_pool = n_pool

        #Down path
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        skips_size = []
        for i in range(self.n_pool):
            block = self._blocktype(
                *args,
                in_conv=in_conv,
                in_meta=in_meta,
                growth_rate = self.growth_rate,
                n_layer = self.n_layer,
                activation=activation,
                kernel_size=kernel_size,
                **kwargs
            )

            in_conv = block.out_conv
            in_meta = block.out_meta

            pool = self._pooltype(2)

            self.down_blocks.append(block)
            self.pools.append(pool)
            skips_size.append(in_conv)

        #Bottleneck
        self.bottleneck = self._blocktype(
            *args,
            in_conv=in_conv,
            in_meta=in_meta,
            activation=activation,
            kernel_size=kernel_size,
            **kwargs
        )

        in_conv = self.bottleneck.out_conv
        in_meta = self.bottleneck.out_meta

        skips_size = skips_size[::-1]

        #Up path
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(self.n_pool):
            up = Interpolate(scale_factor=2)
            in_conv = skips_size[i] + in_conv

            block = self._blocktype(
                *args,
                in_conv=in_conv,
                in_meta=in_meta,
                growth_rate = self.growth_rate,
                n_layer = self.n_layer,
                activation=activation,
                kernel_size=kernel_size,
                **kwargs
            )

            in_conv = block.out_conv
            in_meta = block.out_meta

            self.up_blocks.append(block)
            self.ups.append(up)

        self.last_conv = in_conv
        self.last_meta = in_meta

        self.last_layer_seg = self._layertype(
            *args,
            in_conv=self.last_conv,
            out_conv=self.out_conv_seg,
            in_meta=self.last_meta,
            out_meta=self.out_meta,
            use_global_pooling=False,
            activation=torch.nn.functional.softmax,
            kernel_size=final_kernel_size,
            **kwargs
        )

        self.last_layer_class = self._layertype(
            *args,
            in_conv=self.last_conv,
            out_conv=self.out_conv_class,
            in_meta=self.last_meta,
            out_meta=self.out_meta,
            use_global_pooling=True,
            activation=torch.nn.functional.softmax,
            kernel_size=final_kernel_size,
            **kwargs
        )

        self.last_layer_reg = self._layertype(
            *args,
            in_conv=self.last_conv,
            out_conv=self.out_conv_reg,
            in_meta=self.last_meta,
            out_meta=self.out_meta,
            use_global_pooling=True,
            activation=torch.nn.functional.relu,
            kernel_size=final_kernel_size,
            **kwargs
        )

    def forward(self, img, meta):
        
        #Input
        current_img = img
        current_meta = meta

        #Down path
        skips = []
        for i in range(self.n_pool):
            current_img, current_meta = self.down_blocks[i](current_img, current_meta)
            skips.append(current_img)
            current_img = self.pools[i](current_img)

        #Bottleneck
        current_img, current_meta = self.bottleneck(current_img, current_meta)

        #Up path
        for i, skip in zip(range(self.n_pool), reversed(skips)):
            current_img = self.ups[i](current_img)
            current_img = torch.cat((skip, current_img), 1)
            current_img, current_meta = self.up_blocks[i](current_img, current_meta)

        seg_output = self.last_layer_seg(current_img, current_meta)
        class_output = self.last_layer_class(current_img, current_meta)
        reg_output = self.last_layer_reg(current_img, current_meta)

        return seg_output, class_output, reg_output

class MixedDenseUNet1D(_MixedDenseUNet):
    def __init__(self, *args, pool_type='max', **kwargs):
        self._layertype = get_mixed_block(1)
        self._blocktype = get_mixed_dense_block(1)
        self._pooltype = get_pool(1, pool_type)
        super().__init__(*args, **kwargs)


class MixedDenseUNet2D(_MixedDenseUNet):
    def __init__(self, *args, pool_type='max', **kwargs):
        self._layertype = get_mixed_block(2)
        self._blocktype = get_mixed_dense_block(2)
        self._pooltype = get_pool(2, pool_type)
        super().__init__(*args, **kwargs)


class MixedDenseUNet3D(_MixedDenseUNet):
    def __init__(self, *args, pool_type='max', **kwargs):
        self._layertype = get_mixed_block(3)
        self._blocktype = get_mixed_dense_block(3)
        self._pooltype = get_pool(3, pool_type)
        super().__init__(*args, **kwargs)

class MixedDenseUNetFake3D(_MixedDenseUNet):
    def __init__(self, *args, pool_type='max', **kwargs):
        self._layertype = get_mixed_block(3)
        self._blocktype = get_mixed_dense_block(2.5)
        self._pooltype = get_pool(3, pool_type)
        super().__init__(*args, **kwargs)


_mixed_dense_unets = {
    1: MixedDenseUNet1D,
    2: MixedDenseUNet2D,
    2.5: MixedDenseUNetFake3D,
    3: MixedDenseUNet3D,
}


def get_dense_unet(dim=2):
    if dim not in _mixed_dense_unets:
        raise Exception('Dimensions should be either 1, 2 or 3, not {}'.format(dim))
    return _mixed_dense_unets[dim]
