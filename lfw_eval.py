#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: lfw_eval.py
@time: 2018/12/22 9:47
@desc:
'''

import numpy as np
import scipy.io
import os
import torch.utils.data
from backbone import mobilefacenet
from dataset.lfw import LFW
import torchvision.transforms as transforms
from torch.nn import DataParallel
import argparse

def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def evaluation_10_fold(root='./result/pytorch_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)
    #     print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    # print('--------')
    # print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
    return ACCs

def getFeatureFromTorch(root, file_list, feature_save_dir, gpus='0', resume=None):
    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = mobilefacenet.MobileFaceNet()
    if resume:
        ckpt = torch.load(resume)
        net.load_state_dict(ckpt['net_state_dict'])

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    net.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    lfw_dataset = LFW(root, file_list, transform = transform)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=128,
                                              shuffle=False, num_workers=2, drop_last=False)

    featureLs = None
    featureRs = None
    count = 0

    for data in lfw_loader:
        for i in range(len(data)):
            print(data[i].shape)
            data[i] = data[i].to(device)
        count += data[0].size(0)
        print('extracing deep features from the face pair {}...'.format(count))
        res = [net(d).data.cpu().numpy() for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        # print(featureL.shape, featureR.shape)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # print(featureLs.shape, featureRs.shape)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': lfw_dataset.folds, 'flag': lfw_dataset.flags}
    scipy.io.savemat(feature_save_dir, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--root', type=str, default='/media/ramdisk/lfw_align_112', help='The path of lfw data')
    parser.add_argument('--file_list', type=str, default='/media/ramdisk/pairs.txt', help='The path of lfw data')
    parser.add_argument('--resume', type=str, default='./model/CASIA_v1_20181224_154708/013.ckpt',
                        help='The path pf save model')
    parser.add_argument('--feature_save_dir', type=str, default='./result/tmp_best_result.mat',
                        help='The path of the extract features save, must be .mat file')
    parser.add_argument('--gpus', type=str, default='0,1', help='gpu list')
    args = parser.parse_args()


    getFeatureFromTorch(args.root, args.file_list, args.feature_save_dir, args.gpus, args.resume)
    ACCs = evaluation_10_fold(args.feature_save_dir)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
