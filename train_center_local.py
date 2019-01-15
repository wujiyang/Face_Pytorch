#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: train_center_local.py.py
@time: 2019/1/15 17:37
@desc:
'''

import os
from matplotlib import pyplot as plt
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from backbone.mobilefacenet import MobileFaceNet
from backbone.resnet import ResNet50, ResNet101
from backbone.arcfacenet import SEResNet_IR
from backbone.spherenet import SphereNet
from margin.ArcMarginProduct import ArcMarginProduct
from margin.InnerProduct import InnerProduct
from lossfunctions.centerloss import CenterLoss
from utils.logging import init_log
from dataset.casia_webface import CASIAWebFace
from dataset.lfw import LFW
from torch.optim import lr_scheduler
import torch.optim as optim
import time
from eval_lfw import evaluation_10_fold, getFeatureFromTorch
import numpy as np
import torchvision.transforms as transforms
import argparse

'''plot feature distirbution'''
def plot_features(features, labels, num_classes, epoch, save_dir):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    colors = ['#9ACD32', '#FFFF00', '#000000', '#0000FF', '#F5DEB3',
              '#EE82EE', '#40E0D0', '#FF6347', '#D8BFD8', '#008080',
              '#D2B48C', '#4682B4', '#708090', '#FF0000', '#00FF7F',
              '#8B0000', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD']
    for label_idx in range(num_classes):
        plt.plot(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            '.',
            c=colors[label_idx],
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'], loc='upper right')
    dirname = save_dir
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    save_name = os.path.join(dirname, 'epoch_' + str(epoch) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = os.path.join(args.save_dir, args.model_pre + args.backbone.upper() + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    # validation dataset
    trainset = CASIAWebFace(args.train_root, args.train_file_list, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)
    # test dataset
    lfwdataset = LFW(args.lfw_test_root, args.lfw_file_list, transform=transform)
    lfwloader = torch.utils.data.DataLoader(lfwdataset, batch_size=128,
                                             shuffle=False, num_workers=4, drop_last=False)

    # define backbone and margin layer
    if args.backbone == 'MobileFace':
        net = MobileFaceNet(feature_dim=args.feature_dim)
    elif args.backbone == 'Res50':
        net = ResNet50()
    elif args.backbone == 'Res101':
        net = ResNet101()
    elif args.backbone == 'Res50_IR':
        net = SEResNet_IR(50, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes50_IR':
        net = SEResNet_IR(50, feature_dim=args.feature_dim, mode='se_ir')
    elif args.backbone == 'SphereNet':
        net = SphereNet(num_layers=64, feature_dim=args.feature_dim)
    else:
        print(args.backbone, ' is not available!')

    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'CosFace':
        pass
    elif args.margin_type == 'SphereFace':
        pass
    elif args.margin_type == 'InnerProduct':
        margin = InnerProduct(args.feature_dim, trainset.class_nums)
    else:
        print(args.margin_type, 'is not available!')

    if args.resume:
        print('resume the model parameters from: ', args.net_path, args.margin_path)
        net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
        margin.load_state_dict(torch.load(args.margin_path)['net_state_dict'])

    # define optimizers for different layers
    criterion_classi = torch.nn.CrossEntropyLoss().to(device)
    optimizer_classi = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)

    criterion_center = CenterLoss(trainset.class_nums, args.feature_dim).to(device)
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)

    scheduler_classi = lr_scheduler.MultiStepLR(optimizer_classi, milestones=[35, 60, 85], gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
        margin = DataParallel(margin).to(device)
    else:
        net = net.to(device)
        margin = margin.to(device)

    best_lfw_acc = 0.0
    best_lfw_iters = 0
    total_iters = 0
    for epoch in range(1, args.total_epoch + 1):
        scheduler_classi.step()
        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, args.total_epoch))
        net.train()

        if args.plot:
            all_features, all_labels = [], []

        since = time.time()
        for data in trainloader:
            img, label = data[0].to(device), data[1].to(device)
            feature = net(img)
            output = margin(feature)
            loss_classi = criterion_classi(output, label)
            loss_center = criterion_center(feature, label)
            total_loss = loss_classi + loss_center * args.weight_center

            optimizer_classi.zero_grad()
            optimizer_center.zero_grad()
            total_loss.backward()
            optimizer_classi.step()

            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_center.parameters():
                param.grad.data *= (1. / args.weight_center)
            optimizer_center.step()

            total_iters += 1
            if args.plot:
                feat = feature.data.cpu().numpy()
                #for i in range(feat.shape[0]):
                #    feat[i] = feat[i] / np.sqrt((np.dot(feat[i], feat[i])))
                all_features.append(feat)
                all_labels.append(label.data.cpu().numpy())

            # print train information
            if total_iters % 10 == 0:
                # current training accuracy
                _, predict = torch.max(output.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 10
                since = time.time()
                print("Iters: {:0>6d}/[{:0>2d}], loss_classi: {:.4f}, loss_center: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters,
                                                                                                                                          epoch,
                                                                                                                                          loss_classi.item(),
                                                                                                                                          loss_center.item(),
                                                                                                                                          correct/total,
                                                                                                                                          time_cur,
                                                                                                                                          scheduler_classi.get_lr()[
                                                                                                                                              0]))
            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                    margin_state_dict = margin.state_dict()

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))
                #torch.save({
                #    'iters': total_iters,
                #    'net_state_dict': criterion_center.state_dict()},
                #    os.path.join(save_dir, 'Iter_%06d_center.ckpt' % total_iters))

            # test accuracy
            if total_iters % args.test_freq == 0:

                # test model on lfw
                net.eval()
                getFeatureFromTorch('./result/cur_lfw_result.mat', net, device, lfwdataset, lfwloader)
                lfw_accs = evaluation_10_fold('./result/cur_lfw_result.mat')
                _print('LFW Ave Accuracy: {:.4f}'.format(np.mean(lfw_accs) * 100))
                if best_lfw_acc < np.mean(lfw_accs) * 100:
                    best_lfw_acc = np.mean(lfw_accs) * 100
                    best_lfw_iters = total_iters

                net.train()

        if args.plot:
            all_features = np.concatenate(all_features, 0)
            all_labels = np.concatenate(all_labels, 0)
            plot_features(all_features, all_labels, trainset.class_nums, epoch, save_dir)
    _print('Finally Best Accuracy: LFW: {:.4f} in iters: {}'.format(best_lfw_acc, best_lfw_iters))
    print('finishing training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--train_root', type=str, default='D:/data/webface_align_112', help='train image root')
    parser.add_argument('--train_file_list', type=str, default='D:/data/webface_test.list', help='train list')
    parser.add_argument('--lfw_test_root', type=str, default='D:/data/lfw_align_112', help='lfw image root')
    parser.add_argument('--lfw_file_list', type=str, default='D:/data/pairs.txt', help='lfw pair file list')

    parser.add_argument('--backbone', type=str, default='MobileFace', help='MobileFace, Res50, Res101, Res50_IR, SERes50_IR, SphereNet')
    parser.add_argument('--margin_type', type=str, default='InnerProduct', help='InnerProduct, ArcFace, CosFace, SphereFace')
    parser.add_argument('--feature_dim', type=int, default=2, help='feature dimension, 128 or 512')
    parser.add_argument('--scale_size', type=float, default=32.0, help='scale size')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=100, help='total epochs')
    parser.add_argument('--weight_center', type=float, default=0.1, help='center loss weight')

    parser.add_argument('--save_freq', type=int, default=2000, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=2000, help='test frequency')
    parser.add_argument('--resume', type=int, default=False, help='resume model')
    parser.add_argument('--net_path', type=str, default='', help='resume model')
    parser.add_argument('--margin_path', type=str, default='', help='resume model')
    parser.add_argument('--save_dir', type=str, default='./model', help='model save dir')
    parser.add_argument('--model_pre', type=str, default='Softmax_Center_', help='model prefix')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')
    parser.add_argument('--plot', type=int, default=True, help="whether to plot features for every epoch")

    args = parser.parse_args()

    train(args)


