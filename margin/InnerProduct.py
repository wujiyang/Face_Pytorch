#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: InnerProduct.py
@time: 2019/1/4 16:54
@desc: just normal inner product
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class CosineMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575):
        super(CosineMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input):
        output = F.linear(input, self.weight)
        return output


if __name__ == '__main__':
    pass