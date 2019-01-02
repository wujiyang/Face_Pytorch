#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: plot_theta.py
@time: 2019/1/2 19:08
@desc: plot theta distribution between weight and feature vector
'''

from matplotlib import pyplot as plt
plt.switch_backend('agg')

import argparse
from backbone.mobilefacenet import MobileFaceNet
from margin.ArcMarginProduct import ArcMarginProduct
from torch.utils.data import DataLoader
import torch
from torch.nn import DataParallel

from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
from dataset.casia_webface import CASIAWebFace


def get_train_loader(img_folder, filelist):
    print('Loading dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    trainset = CASIAWebFace(img_folder, filelist, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=False, num_workers=8, drop_last=False)
    return trainloader

def load_model(backbone_state_dict, margin_state_dict, device):

    # load model
    net = MobileFaceNet()
    net.load_state_dict(torch.load(backbone_state_dict)['net_state_dict'])
    margin = ArcMarginProduct(in_feature=128, out_feature=10575)
    margin.load_state_dict(torch.load(margin_state_dict)['net_state_dict'])

    net = net.to(device)
    margin = margin.to(device)

    return net.eval(), margin.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot theta distribution of trained model')
    parser.add_argument('--img_root', type=str, default='/media/ramdisk/webface_align_112', help='train image root')
    parser.add_argument('--file_list', type=str, default='/media/ramdisk/webface_align_train.list', help='train list')
    parser.add_argument('--backbone_file', type=str, default='../model/CASIA_MOBILEFACE_20190102_155410/Iter_052500_net.ckpt', help='backbone state dict file')
    parser.add_argument('--margin_file', type=str, default='../model/CASIA_MOBILEFACE_20190102_155410/Iter_052500_margin.ckpt', help='backbone state dict file')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix, single gpu only')
    args = parser.parse_args()

    # gpu init
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    net, margin = load_model(args.backbone_file, args.margin_file, device)

    image_loader = get_train_loader(args.img_root, args.file_list)
    theta = []
    for data in image_loader:
        img, label = data[0].to(device), data[1].to(device)
        embedding = net(img)
        cos_theta = F.linear(F.normalize(embedding), F.normalize(margin.weight))
        cos_theta = cos_theta.clamp(-1, 1).detach().cpu().numpy()
        for i in range(img.shape[0]):
            cos_trget = cos_theta[i][label[i]]
            theta.append(np.arccos(cos_trget) / np.pi * 180)

    # initial model
    net = MobileFaceNet()
    margin = ArcMarginProduct()
    net = net.to(device).eval()
    margin = margin.to(device).eval()
    theta_initial = []
    for data in image_loader:
        img, label = data[0].to(device), data[1].to(device)
        # print(img.shape, label.shape)
        embedding = net(img)
        cos_theta = F.linear(F.normalize(embedding), F.normalize(margin.weight))
        cos_theta = cos_theta.clamp(-1, 1).detach().cpu().numpy()
        for i in range(img.shape[0]):
            cos_trget = cos_theta[i][label[i]]
            theta_initial.append(np.arccos(cos_trget) / np.pi * 180)

    # plot the theta
    plt.figure()
    plt.xlabel('theta distribution of images')
    plt.ylabel('count')
    plt.title('theta_hist')
    plt.hist(theta, bins=60, normed=0)
    plt.hist(theta_initial, bins=60, normed=0)
    plt.savefig('theta_distribution_hist.jpg')
    # plot the initial theta
