#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: eval_lfw_blufr.py
@time: 2019/1/17 15:52
@desc: test lfw accuracy on blufr protocol
'''
'''
LFW BLUFR TEST PROTOCOL

Official Website: http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/

When I try to do this, I find that the blufr_lfw_config.mat file provided by above site is too old.
Some image files listed in the mat have been removed in lfw pairs.txt
So this work is suspended for now...
'''

import scipy.io as sio
import argparse

def readName(file='pairs.txt'):
    name_list = []
    f = open(file, 'r')
    lines = f.readlines()

    for line in lines[1:]:
        line_split = line.rstrip().split()
        if len(line_split) == 3:
            name_list.append(line_split[0])
        elif len(line_split) == 4:
            name_list.append(line_split[0])
            name_list.append(line_split[2])
        else:
            print('wrong file, please check again')

    return list(set(name_list))


def main(args):
    blufr_info = sio.loadmat(args.lfw_blufr_file)
    #print(blufr_info)
    name_list = readName()

    image = blufr_info['imageList']
    missing_files = []
    for i in range(image.shape[0]):
        name = image[i][0][0]
        index = name.rfind('_')
        name = name[0:index]
        if name not in name_list:
            print(name)
            missing_files.append(name)
    print('lfw pairs.txt total persons: ', len(name_list))
    print('blufr_mat_missing persons: ', len(missing_files))

    '''
    Some of the missing file:
    Zdravko_Mucic
    Zelma_Novelo
    Zeng_Qinghong
    Zumrati_Juma
    lfw pairs.txt total persons:  4281
    blufr_mat_missing persons:  1549
    
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lfw blufr test')
    parser.add_argument('--lfw_blufr_file', type=str, default='./blufr_lfw_config.mat', help='feature dimension')
    parser.add_argument('--lfw_pairs.txt', type=str, default='./pairs.txt', help='feature dimension')
    parser.add_argument('--gpus', type=str, default='2,3', help='gpu list')
    args = parser.parse_args()

    main(args)