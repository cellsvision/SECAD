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


class FMC(nn.Module):
    def __init__(self,class_num=6, inplanes=2048, stride=1, embedding_size=512):
        super(FMC, self).__init__()


        self.conv0 = nn.Conv2d(inplanes, embedding_size, kernel_size=1, stride=stride, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.ca0 = ChannelAttention(embedding_size)
        self.sa0 = SpatialAttention()

        self.conv1 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embedding_size)
        self.ca1 = ChannelAttention(embedding_size)
        self.sa1 = SpatialAttention()

        self.conv2 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(embedding_size)
        self.ca2 = ChannelAttention(embedding_size)
        self.sa2 = SpatialAttention()

        self.conv3 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3, stride=stride,  bias=False)
        self.bn3 = nn.BatchNorm2d(embedding_size)
        self.ca3 = ChannelAttention(embedding_size)
        self.sa3 = SpatialAttention()

        self.conv4 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3, stride=stride,  bias=False)
        self.bn4 = nn.BatchNorm2d(embedding_size)
        self.ca4 = ChannelAttention(embedding_size)
        self.sa4 = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embedding_size, embedding_size),
            nn.Dropout(0.5),
            Mish(),
            nn.Linear(embedding_size, 512),
            nn.Dropout(0.4),
            Mish(),
            nn.Linear(512, class_num),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # residual = x
        out = self.conv0(x)
        out = self.ca0(out) * out
        out = self.sa0(out) * out


        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ca1(out) * out
        out = self.sa1(out) * out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.ca2(out) * out
        out = self.sa2(out) * out


        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.ca3(out) * out
        out = self.sa3(out) * out

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.ca4(out) * out
        out = self.sa4(out) * out
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc(out)

        return out


if __name__=='__main__':
    model = FMC()
    input = torch.zeros((2,2048,56,60))
    output = model(input)
    print(output)