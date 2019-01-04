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
    parser.add_argument('--backbone_file', type=str, default='../model/Paper_MOBILEFACE_20190103_111830/Iter_088000_net.ckpt', help='backbone state dict file')
    parser.add_argument('--margin_file', type=str, default='../model/Paper_MOBILEFACE_20190103_111830/Iter_088000_margin.ckpt', help='backbone state dict file')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix, single gpu only')
    args = parser.parse_args()

    # gpu init
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load pretrain model
    trained_net, trained_margin = load_model(args.backbone_file, args.margin_file, device)

    # initial model
    initial_net = MobileFaceNet()
    initial_margin = ArcMarginProduct()
    initial_net = initial_net.to(device).eval()
    initial_margin = initial_margin.to(device).eval()

    # image dataloader
    image_loader = get_train_loader(args.img_root, args.file_list)
    theta_trained = []
    theta_initial = []
    for data in image_loader:
        img, label = data[0].to(device), data[1].to(device)
        # pretrained
        embedding = trained_net(img)
        cos_theta = F.linear(F.normalize(embedding), F.normalize(trained_margin.weight))
        cos_theta = cos_theta.clamp(-1, 1).detach().cpu().numpy()
        for i in range(img.shape[0]):
            cos_trget = cos_theta[i][label[i]]
            theta_trained.append(np.arccos(cos_trget) / np.pi * 180)
        # initial
        embedding = initial_net(img)
        cos_theta = F.linear(F.normalize(embedding), F.normalize(initial_margin.weight))
        cos_theta = cos_theta.clamp(-1, 1).detach().cpu().numpy()
        for i in range(img.shape[0]):
            cos_trget = cos_theta[i][label[i]]
            theta_initial.append(np.arccos(cos_trget) / np.pi * 180)
    '''
    # write theta list to txt file
    trained_theta_file = open('arcface_theta.txt', 'w')
    initial_theta_file = open('initial_theta.txt', 'w')
    for item in theta_trained:
        trained_theta_file.write(str(item))
        trained_theta_file.write('\n')
    for item in theta_initial:
        initial_theta_file.write(str(item))
        initial_theta_file.write('\n')

    # plot the theta, read theta from txt first
    theta_trained = []
    theta_initial = []
    trained_theta_file = open('arcface_theta.txt', 'r')
    initial_theta_file = open('initial_theta.txt', 'r')
    lines = trained_theta_file.readlines()
    for line in lines:
        theta_trained.append(float(line.strip('\n')[0]))
    lines = initial_theta_file.readlines()
    for line in lines:
        theta_initial.append(float(line.split('\n')[0]))
    '''
    print(len(theta_trained), len(theta_initial))
    plt.figure()
    plt.xlabel('Theta')
    plt.ylabel('Numbers')
    plt.title('Theta Distribution')
    plt.hist(theta_trained, bins=180, normed=0)
    plt.hist(theta_initial, bins=180, normed=0)
    plt.legend(['trained theta distribution', 'initial theta distribution'])
    plt.savefig('theta_distribution_hist.jpg')
