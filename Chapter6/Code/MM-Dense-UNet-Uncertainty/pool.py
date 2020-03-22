import torch.nn as nn


class _GlobalMaxPool1d(nn.AdaptiveMaxPool1d):
    def __init__(self):
        super().__init__((1))

    def forward(self, x):
        return super().forward(x)


class _GlobalMaxPool2d(nn.AdaptiveMaxPool2d):
    def __init__(self):
        super().__init__((1, 1))

    def forward(self, x):
        return super().forward(x)


class _GlobalMaxPool3d(nn.AdaptiveMaxPool3d):
    def __init__(self):
        super().__init__((1, 1, 1))

    def forward(self, x):
        return super().forward(x)

class _GlobalAvgPool1d(nn.AdaptiveAvgPool1d):
    def __init__(self):
        super().__init__((1))

    def forward(self, x):
        return super().forward(x)


class _GlobalAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self):
        super().__init__((1, 1))

    def forward(self, x):
        return super().forward(x)


class _GlobalAvgPool3d(nn.AdaptiveAvgPool3d):
    def __init__(self):
        super().__init__((1, 1, 1))

    def forward(self, x):
        return super().forward(x)


_max_pools = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
_avg_pools = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}
_global_max_pools = {1: _GlobalMaxPool1d, 2: _GlobalMaxPool2d, 3: _GlobalMaxPool3d}
_global_avg_pools = {1: _GlobalAvgPool1d, 2: _GlobalAvgPool2d, 3: _GlobalAvgPool3d}
_pools = {'max': _max_pools, 'avg': _avg_pools, 'global_max': _global_max_pools, 'global_avg': _global_avg_pools}


def get_pool(dim, sort='max'):
    """N-dimension Getter for Pooling Layers.

    :param dim: Dimension of the Pooling layer wanted.
    :type dim: int
    :param sort: Sort of Pooling wanted. ``max``, ``avg`` or ``global_max``.
                 ``max`` by default.
    :type sort: string
    :return: A Pooling layer Constructor in the right dimension.
    """
    if sort not in _pools:
        raise Exception("Expected 'max' or 'avg' for pooling. Got {}".format(sort))

    sortpool = _pools[sort]
    if dim not in sortpool:
        raise Exception('Dimensions should be either 1, 2 or 3, not {}'.format(dim))

    return sortpool[dim]
