#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: centerloss.py
@time: 2019/1/4 15:24
@desc: the implementation of center loss
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.Tensor(self.num_classes, self.feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x, label):
        '''
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        '''
        batch_size = x.size(0)
        # compute the distance of (x-center)^2
        dis = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        dis = torch.addmm(1, dis, -2, x, self.centers.t())

        # get one_hot matrix
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classes = torch.arange(self.num_classes).long().to(device)
        label = label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = label.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = dis[i][mask[i]]
            value = value.clamp(min=1e-12, max =1e-12)
            dist.append(value)

        dist = torch.cat(dist)
        loss = dist.mean()

        return loss