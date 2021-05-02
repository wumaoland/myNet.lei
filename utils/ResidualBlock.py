from typing import Optional, Callable, List

import torch
from torch import nn, Tensor

from utils.BasicConv2d import BasicConv2d
from utils.Inception import Inception
import torch.nn.functional as F


class ResidualBlock2(nn.Module):
    def __init__(
            self,
            params1: List[int],
            params2: List[int],
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        if conv_block is None:
            conv_block = BasicConv2d

        super(ResidualBlock2, self).__init__()
        self.inception1 = Inception(params1[0], params1[1], params1[2], params1[3],
                                    params1[4], params1[5], params1[6], conv_block)
        out1 = params1[1] + params1[3] + params1[5] + params1[6]
        self.inception2 = Inception(out1, params2[0], params2[1], params2[2],
                                    params2[3], params2[4], params2[5], conv_block)

        self.downsample = conv_block(params2[0] + params2[2] + params2[4] + params2[5], params1[0], kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.downsample(x)

        # Don't use x += residual, as it would cause a runtime error
        x = x + residual
        x = self.relu(x)

        return x


class UpSample(nn.Module):
    def __init__(
            self,
            params1: List[int],
            params2: List[int],
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(UpSample, self).__init__()

        assert len(params1) == 7
        assert len(params2) == 6

        if conv_block is None:
            conv_block = BasicConv2d

        self.inception1 = Inception(params1[0], params1[1], params1[2], params1[3],
                                    params1[4], params1[5], params1[6], conv_block)
        out1 = params1[1] + params1[3] + params1[5] + params1[6]
        self.inception2 = Inception(out1, params2[0], params2[1], params2[2],
                                    params2[3], params2[4], params2[5], conv_block)
        self.maxPool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxPool(x)

        return x
