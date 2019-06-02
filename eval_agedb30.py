#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: eval_agedb30.py
@time: 2018/12/25 19:05
@desc: The AgeDB-30 test protocol is same with LFW, so I just copy the code from eval_lfw.py
'''


import numpy as np
import scipy.io
import os
import torch.utils.data
from backbone import mobilefacenet, resnet, arcfacenet, cbam
from dataset.agedb import AgeDB30
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

def evaluation_10_fold(feature_path='./result/cur_epoch_agedb_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(feature_path)
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

    return ACCs

def loadModel(data_root, file_list, backbone_net, gpus='0', resume=None):

    if backbone_net == 'MobileFace':
        net = mobilefacenet.MobileFaceNet()
    elif backbone_net == 'CBAM_50':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_50_SE':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif backbone_net == 'CBAM_100':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_100_SE':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    else:
        print(backbone_net, ' is not available!')

    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(resume)['net_state_dict'])

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    agedb_dataset = AgeDB30(data_root, file_list, transform=transform)
    agedb_loader = torch.utils.data.DataLoader(agedb_dataset, batch_size=128,
                                             shuffle=False, num_workers=2, drop_last=False)

    return net.eval(), device, agedb_dataset, agedb_loader

def getFeatureFromTorch(feature_save_dir, net, device, data_set, data_loader):
    featureLs = None
    featureRs = None
    count = 0
    for data in data_loader:
        for i in range(len(data)):
            data[i] = data[i].to(device)
        count += data[0].size(0)
        #print('extracing deep features from the face pair {}...'.format(count))
        with torch.no_grad():
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

    result = {'fl': featureLs, 'fr': featureRs, 'fold': data_set.folds, 'flag': data_set.flags}
    scipy.io.savemat(feature_save_dir, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--root', type=str, default='/media/sda/AgeDB-30/agedb30_align_112', help='The path of lfw data')
    parser.add_argument('--file_list', type=str, default='/media/sda/AgeDB-30/agedb_30_pair.txt', help='The path of lfw data')
    parser.add_argument('--resume', type=str, default='./model/SERES100_SERES100_IR_20190528_132635/Iter_342000_net.ckpt', help='The path pf save model')
    parser.add_argument('--backbone_net', type=str, default='CBAM_100_SE', help='MobileFace, CBAM_50, CBAM_50_SE, CBAM_100, CBAM_100_SE')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension')
    parser.add_argument('--feature_save_path', type=str, default='./result/cur_epoch_agedb_result.mat',
                        help='The path of the extract features save, must be .mat file')
    parser.add_argument('--gpus', type=str, default='2,3', help='gpu list')
    args = parser.parse_args()

    net, device, agedb_dataset, agedb_loader = loadModel(args.root, args.file_list, args.backbone_net, args.gpus, args.resume)
    getFeatureFromTorch(args.feature_save_path, net, device, agedb_dataset, agedb_loader)
    ACCs = evaluation_10_fold(args.feature_save_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.4f}'.format(np.mean(ACCs) * 100))

