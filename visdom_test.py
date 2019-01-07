#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: visdom_test.py
@time: 2019/1/7 15:05
@desc:
'''

import time

from utils.visualize import Visualizer
vis = Visualizer(env='test')
for i in range(20):
    x = i
    y = 2 * i
    vis.plot_curves({'train': x, 'test': y}, iters=i)
    time.sleep(1)