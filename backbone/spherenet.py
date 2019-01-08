#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: spherenet.py
@time: 2018/12/26 10:14
@desc: A 64 layer residual network struture used in sphereface and cosface, for fast convergence, I add BN after every Conv layer.
'''

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU(channels)

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)

        return x + short_cut


class SphereNet(nn.Module):
    def __init__(self, num_layers = 20, feature_dim=512):
        super(SphereNet, self).__init__()
        assert num_layers in [20, 64], 'SphereNet num_layers should be 20 or 64'
        if num_layers == 20:
            layers = [1, 2, 4, 1]
        elif num_layers == 64:
            layers = [3, 7, 16, 3]
        else:
            raise ValueError('sphere' + str(num_layers) + " IS NOT SUPPORTED! (sphere20 or sphere64)")

        filter_list = [3, 64, 128, 256, 512]
        block = Block
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.fc = nn.Linear(512 * 7 * 7, feature_dim)
        self.last_bn = nn.BatchNorm1d(feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)

    def _make_layer(self, block, inplanes, planes, num_units, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.PReLU(planes))
        for i in range(num_units):
            layers.append(block(planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.last_bn(x)

        return x


if __name__ == '__main__':
    input = torch.Tensor(2, 3, 112, 112)
    net = SphereNet(num_layers=64, feature_dim=512)

    out = net(input)
    print(out.shape)

