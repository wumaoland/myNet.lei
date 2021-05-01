from typing import Optional, Callable, List

import torch
from torch import nn, Tensor

from utils.BasicConv2d import BasicConv2d
from utils.Inception import Inception
import torch.nn.functional as F


# 单独提取出残渣网络中的一层，方便后续处理
class ResidualBlock1(nn.Module):
    def __init__(self):
        super(ResidualBlock1, self).__init__()
        self.inception1a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception1b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.maxpool3(x)

        return x


class ResidualBlock2(nn.Module):
    def __init__(
            self,
            params1: List[int],
            params2: List[int],
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResidualBlock2, self).__init__()
        # assert params2[0] == params1[1] + params1[3] + params1[5] + params1[6]
        self.inception1 = Inception(params1[0], params1[1], params1[2], params1[3],
                                    params1[4], params1[5], params1[6], conv_block)
        out1 = params1[1] + params1[3] + params1[5] + params1[6]
        self.inception2 = Inception(out1, params2[0], params2[1], params2[2],
                                    params2[3], params2[4], params2[5], conv_block)

        self.downsample = BasicConv2d(params2[0] + params2[2] + params2[4] + params2[5], params1[0], kernel_size=1)
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


class ResidualBlock3(nn.Module):
    def __init__(self):
        super(ResidualBlock3, self).__init__()
        self.inception1a = Inception(480, 112, 144, 288, 32, 64, 64)
        self.inception1b = Inception(528, 256, 160, 320, 32, 128, 128)
        self.inception2a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception2b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.downsample = BasicConv2d(1024, 832, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.maxpool4(x)

        residual = x
        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.downsample(x)
        x = x + residual
        x = self.relu(x)
        x = self.avgpool(x)

        return x
