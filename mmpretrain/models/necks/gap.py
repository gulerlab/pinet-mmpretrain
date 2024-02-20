# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalAveragePooling, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
                                 f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs


@MODELS.register_module()
class GlobalAveragePoolingConv2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePoolingConv2d, self).__init__()

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = []
            for x in inputs:
                tmp = x.view(x.shape[0], x.shape[1], -1)
                tmp = (tmp @ torch.ones(tmp.shape[-1], device=tmp.device)) / tmp.shape[-1]
                outs.append(tmp.view(tmp.shape[0], -1))
            outs = tuple(outs)
        elif isinstance(inputs, torch.Tensor):
            tmp = inputs.view(inputs.shape[0], inputs.shape[1], -1)
            tmp = (tmp @ torch.ones(tmp.shape[-1], device=tmp.device)) / tmp.shape[-1]
            outs = tmp.view(tmp.shape[0], -1)
            outs = tuple([outs])
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
