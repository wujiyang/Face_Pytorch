#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: SphereMarginProduct.py
@time: 2018/12/25 9:19
@desc: multiplicative angular margin for sphereface
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class SphereMarginProduct(nn.Module):
    def __init__(self, in_feature, out_feature, m=4, base=1000.0, gamma=0.0001, power=2, lambda_min=5.0, iter=0):
        assert m in [1, 2, 3, 4], 'margin should be 1, 2, 3 or 4'
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.m = m
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min
        self.iter = 0
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.margin_formula = [
            lambda x : x ** 0,
            lambda x : x ** 1,
            lambda x : 2 * x ** 2 - 1,
            lambda x : 4 * x ** 3 - 3 * x,
            lambda x : 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x : 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        self.iter += 1
        self.cur_lambda = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta(-1, 1)

        cos_m_theta = self.margin_formula(self.m)(cos_theta)
        theta = cos_theta.data.acos()
        k = ((self.m * theta) / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        phi_theta_ = (self.cur_lambda * cos_theta + phi_theta) / (1 + self.cur_lambda)
        norm_of_feature = torch.norm(input, 2, 1)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = one_hot * phi_theta_ + (1 - one_hot) * cos_theta
        output *= norm_of_feature.view(-1, 1)

        return output


if __name__ == '__main__':
    pass