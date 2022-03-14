# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.weight_init import init_net_weights


class Net(nn.Module):
    """
    Build a general Net models with a list of blocks for video recognition.

    ::

                                         Input
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓

    The ResNet builder can be found in `create_resnet`.
    """

    def __init__(self, *, blocks: nn.ModuleList) -> None:
        """
        Args:
            blocks (torch.nn.module_list): the list of block modules.
        """
        super().__init__()
        assert blocks is not None
        self.blocks = blocks
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        init_net_weights(self)

    def forward(self, x: torch.Tensor, emb_layer=1, path=None, stage=None, fuse=None) -> torch.Tensor:
        for idx in range(len(self.blocks)):
            if idx == len(self.blocks) - emb_layer: # idx = 6
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x
            else:
                x = self.blocks[idx](x)
                if path:
                    if not stage:
                        if idx == 5:
                            # x_slow = x[:,:2048,:,:,:]
                            x_slow = x[0]
                            x_slow = nn.AvgPool3d((x_slow.shape[2], x_slow.shape[3], x_slow.shape[4]))(x_slow)
                            x_slow = x_slow.view(x_slow.size(0), -1)
                            # x_fast = x[:,2048:,:,:,:]
                            x_fast = x[1]
                            x_fast = nn.AvgPool3d((x_fast.shape[2], x_fast.shape[3], x_fast.shape[4]))(x_fast)
                            x_fast = x_fast.view(x_fast.size(0), -1)
                            if path == 'cat':
                                x = [x_slow, x_fast]
                                x = torch.cat(x, dim=1)
                            elif path == 'slow':
                                x = x_slow
                            elif path == 'fast':
                                x = x_fast
                            return x
                    else:
                        if idx == stage-1:
                            x_slow = x[0]
                            x_slow = nn.AvgPool3d((x_slow.shape[2], x_slow.shape[3], x_slow.shape[4]))(x_slow)
                            x_slow = x_slow.view(x_slow.size(0), -1)
                            x_fast = x[1]
                            x_fast = nn.AvgPool3d((x_fast.shape[2], x_fast.shape[3], x_fast.shape[4]))(x_fast)
                            x_fast = x_fast.view(x_fast.size(0), -1)
                            if path == 'cat':
                                x = [x_slow, x_fast]
                                x = torch.cat(x, dim=1)
                            elif path == 'slow':
                                x = x_slow
                            elif path == 'fast':
                                x = x_fast
                            return x
                else:
                    if not stage:
                        if idx == 4:
                            x = nn.AvgPool3d((x.shape[2], x.shape[3], x.shape[4]))(x)
                            x = x.view(x.size(0), -1)
                            return x
                    else:
                        if idx == stage - 1:
                            x = nn.AvgPool3d((x.shape[2], x.shape[3], x.shape[4]))(x)
                            x = x.view(x.size(0), -1)
                            return x

        return x


class MultiPathWayWithFuse(nn.Module):
    """
    Build multi-pathway block with fusion for video recognition, each of the pathway
    contains its own Blocks and Fusion layers across different pathways.

    ::

                            Pathway 1  ... Pathway N
                                ↓              ↓
                             Block 1        Block N
                                ↓⭠ --Fusion----↓
    """

    def __init__(
        self,
        *,
        multipathway_blocks: nn.ModuleList,
        multipathway_fusion: Optional[nn.Module],
        inplace: Optional[bool] = True,
    ) -> None:
        """
        Args:
            multipathway_blocks (nn.module_list): list of models from all pathways.
            multipathway_fusion (nn.module): fusion model.
            inplace (bool): If inplace, directly update the input list without making
                a copy.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        assert isinstance(
            x, list
        ), "input for MultiPathWayWithFuse needs to be a list of tensors"
        if self.inplace:
            x_out = x
        else:
            x_out = [None] * len(x)
        for pathway_idx in range(len(self.multipathway_blocks)):
            if self.multipathway_blocks[pathway_idx] is not None:
                x_out[pathway_idx] = self.multipathway_blocks[pathway_idx](
                    x[pathway_idx]
                )
        if self.multipathway_fusion is not None:
            x_out = self.multipathway_fusion(x_out)
        return x_out




class Distribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean = torch.chunk(parameters, 1, dim=1)
        self.deterministic = deterministic

    def sample(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = self.mean + self.std*torch.randn(self.mean.shape).to(device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5*torch.sum(torch.pow(self.mean, 2)
                        + self.var - 1.0 - self.logvar,
                        dim=[1,2,3])
            else:
                return 0.5*torch.sum(
                        torch.pow(self.mean - other.mean, 2) / other.var
                        + self.var / other.var - 1.0 - self.logvar + other.logvar,
                        dim=[1,2,3])

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0*np.pi)
        return 0.5*torch.sum(
                logtwopi+self.logvar+torch.pow(sample-self.mean, 2) / self.var,
                dim=[1,2,3])

    def mode(self):
        return self.mean