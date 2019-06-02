#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: megaface.py
@time: 2018/12/24 16:29
@desc:
'''

import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)


class MegaFace(data.Dataset):
    def __init__(self, facescrub_dir, megaface_dir, transform=None, loader=img_loader):

        self.transform = transform
        self.loader = loader

        test_image_file_list = []
        print('Scanning files under facescrub and megaface...')
        for root, dirs, files in os.walk(facescrub_dir):
            for e in files:
                filename = os.path.join(root, e)
                ext = os.path.splitext(filename)[1].lower()
                if ext in ('.png', '.bmp', '.jpg', '.jpeg'):
                    test_image_file_list.append(filename)
        for root, dirs, files in os.walk(megaface_dir):
            for e in files:
                filename = os.path.join(root, e)
                ext = os.path.splitext(filename)[1].lower()
                if ext in ('.png', '.bmp', '.jpg', '.jpeg'):
                    test_image_file_list.append(filename)

        self.image_list = test_image_file_list

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = self.loader(img_path)

        #水平翻转图像
        #img = cv2.flip(img, 1)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        return img, img_path

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    facescrub = '/media/sda/megaface_test_kit/facescrub_align_112/'
    megaface = '/media/sda/megaface_test_kit/megaface_align_112/'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    dataset = MegaFace(facescrub, megaface, transform=transform)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data.shape)