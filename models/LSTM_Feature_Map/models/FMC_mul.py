import sys
import os
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from models.mish import Mish


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, activation=nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.activation = activation
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.activation(out)

        return out


class FMC(nn.Module):
    def __init__(self, class_num=6, inplanes=2048, stride=1, inplane_list=[512, 512, 512, 512], embedding_size_list=[512, 512, 512],
                 activation_list=['relu', 'relu', 'relu', 'mish', 'silu', 'silu'], dropout_list=[0.5, 0.5, 0.5]):
        super(FMC, self).__init__()

        self.conv0 = nn.Conv2d(inplanes, inplane_list[0], kernel_size=1, stride=stride, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.ca0 = ChannelAttention(inplane_list[0])
        self.sa0 = SpatialAttention()
        layers = []
        activation_dict = {
            'relu': nn.ReLU(inplace=True),
            'mish': Mish(),
            'silu': nn.SiLU(inplace=True),
            'gelu': nn.GELU(),
        }
        for i in range(len(inplane_list) - 1):
            activation = activation_dict[activation_list[i]]
            layer = BasicBlock(inplane_list[i], inplane_list[i + 1], activation=activation)
            layers.append(layer)
        self.conv_layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout0 = nn.Dropout(0.5)
        fc_layers = []
        embedding_size_list.insert(0, inplane_list[-1])
        for i in range(len(embedding_size_list) - 1):
            activation = activation_dict[activation_list[len(inplane_list) + i - 1]]
            layer = nn.Sequential(
                nn.Linear(embedding_size_list[i], embedding_size_list[i + 1]),
                nn.Dropout(dropout_list[i]),
                activation
            )
            fc_layers.append(layer)
        self.fc = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(embedding_size_list[-1], class_num)

    def forward(self, x):
        # residual = x
        out = self.conv0(x)
        out = self.ca0(out) * out
        out = self.sa0(out) * out

        out = self.conv_layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout0(out)
        out = self.fc(out)
        out = self.output_layer(out)

        return out


if __name__ == '__main__':
    model = FMC()
    input = torch.zeros((2, 2048, 56, 60))
    output = model(input)
    print(output)
