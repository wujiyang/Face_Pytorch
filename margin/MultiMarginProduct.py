#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: MultiMarginProduct.py
@time: 2019/3/30 10:09
@desc: Combination of additive angular margin and additive cosine margin
'''

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class MultiMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m1=0.20, m2=0.35, easy_margin=False):
        super(MultiMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m1 = math.cos(m1)
        self.sin_m1 = math.sin(m1)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m1)
        self.mm = math.sin(math.pi - m1) * m1

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m1 - sine * self.sin_m1

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)


        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) # additive angular margin
        output = output - one_hot * self.m2 # additive cosine margin
        output = output * self.s

        return output


if __name__ == '__main__':
    pass