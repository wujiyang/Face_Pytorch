#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: plot_logit.py
@time: 2019/3/29 14:21
@desc: plot the logit corresponding to shpereface, cosface, arcface and so on.
'''

import math
import torch
import matplotlib.pyplot as plt
import numpy as np

def softmax(theta):
    return torch.cos(theta)

def sphereface(theta, m=4):
    return (torch.cos(m * theta) + 20 * torch.cos(theta)) / (20 + 1)

def cosface(theta, m):
    return torch.cos(theta) - m

def arcface(theta, m):
    return torch.cos(theta + m)

def multimargin(theta, m1, m2):
    return torch.cos(theta + m1) - m2


theta = torch.arange(0, math.pi, 0.001)
print(theta.type)

x = theta.numpy()
y_softmax = softmax(theta).numpy()
y_cosface = cosface(theta, 0.35).numpy()
y_arcface = arcface(theta, 0.5).numpy()

y_multimargin_1 = multimargin(theta, 0.2, 0.3).numpy()
y_multimargin_2 = multimargin(theta, 0.2, 0.4).numpy()
y_multimargin_3 = multimargin(theta, 0.3, 0.2).numpy()
y_multimargin_4 = multimargin(theta, 0.3, 0.3).numpy()
y_multimargin_5 = multimargin(theta, 0.4, 0.2).numpy()
y_multimargin_6 = multimargin(theta, 0.4, 0.3).numpy()

plt.plot(x, y_softmax, x, y_cosface, x, y_arcface, x, y_multimargin_1, x, y_multimargin_2, x, y_multimargin_3, x, y_multimargin_4, x, y_multimargin_5, x, y_multimargin_6)
plt.legend(['Softmax(0.00, 0.00)', 'CosFace(0.00, 0.35)', 'ArcFace(0.50, 0.00)', 'MultiMargin(0.20, 0.30)', 'MultiMargin(0.20, 0.40)', 'MultiMargin(0.30, 0.20)', 'MultiMargin(0.30, 0.30)', 'MultiMargin(0.40, 0.20)', 'MultiMargin(0.40, 0.30)'])
plt.grid(False)
plt.xlim((0, 3/4*math.pi))
plt.ylim((-1.2, 1.2))

plt.xticks(np.arange(0, 2.4, 0.3))
plt.yticks(np.arange(-1.2, 1.2, 0.2))
plt.xlabel('Angular between the Feature and Target Center (Radian: 0 - 3/4 Pi)')
plt.ylabel('Target Logit')

plt.savefig('target logits')