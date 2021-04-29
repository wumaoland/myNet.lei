from typing import Optional, Callable, List

import torch
from torch import nn, Tensor

from utils.Inception import Inception
import torch.nn.functional as F


# 单独提取出残渣网络中的一层，方便后续处理
class ResidualBlock1(nn.Module):
    def __init__(self):
        super(ResidualBlock1, self).__init__()
        self.inception1a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception1b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.inception2a = Inception(480, 192, 96, 208, 16, 48, 64)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.maxpool3(x)
        x = self.inception2a(x)

        return x


class ResidualBlock2(nn.Module):
    def __init__(
            self,
            params: List[int],
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResidualBlock2, self).__init__()
        assert params[0] == params[1] + params[3] + params[5] + params[6]
        self.inception = Inception(params[0], params[1], params[2], params[3],
                                   params[4], params[5], params[6], conv_block)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.inception(x)

        x += residual
        x = self.relu(x)

        return x


class ResidualBlock3(nn.Module):
    def __init__(self):
        super(ResidualBlock3, self).__init__()
        self.inception1a = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception1b = Inception(528, 256, 160, 320, 32, 128, 128)
        self.inception2a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception2b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.maxpool4(x)

        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.avgpool(x)

        return x
