import warnings

import torch
import torch.nn as nn
from typing import Optional, List, Callable, Type

from torch import Tensor

from utils import ResidualBlock
from utils.BasicConv2d import BasicConv2d


class RegoNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            init_channel: int = 3,
            layers=None,
            transform_input: bool = False,
            init_weights: Optional[bool] = None,
    ) -> None:
        super(RegoNet, self).__init__()
        if layers is None:
            layers = [2, 2]
        if init_weights is None:
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True

        self.transform_input = transform_input

        self.conv1 = BasicConv2d(init_channel, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.layer1 = ResidualBlock.ResidualBlock1()
        params1 = [512, 160, 112, 224, 24, 64, 64]
        self.layer2 = self._make_layer(ResidualBlock.ResidualBlock2, params1, layers[0])
        params2 = [512, 128, 128, 256, 24, 64, 64]
        self.layer3 = self._make_layer(ResidualBlock.ResidualBlock2, params2, layers[1])
        self.layer4 = ResidualBlock.ResidualBlock3()

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _make_layer(
            self,
            block: Type[ResidualBlock.ResidualBlock2],
            params: List[int],
            blocks: int = 1,
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> nn.Sequential:
        assert len(params) == 7

        layers = []
        for _ in range(0, blocks):
            layers.append(block(params, conv_block))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 512 x 14 x 14
        x = self.layer1(x)
        # N x 512 x 14 x 14
        x = self.layer2(x)
        # N x 512 x 14 x 14
        x = self.layer3(x)
        # N x 1024 x 7 x 7
        x = self.layer4(x)

        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
