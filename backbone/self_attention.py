#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: self_attention.py
@time: 2019/2/15 19:53
@desc: self residual attention for deep face recognition, my own paper.
'''
import torch
import torch.nn as nn
import time

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class NaiveChannelAttentionModule(nn.Module):
    def __init__(self, channels):
        super(NaiveChannelAttentionModule, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels),
                                    nn.PReLU(channels))
        self.conv_2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels),
                                    nn.PReLU(channels))
        self.conv_3 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels),
                                    nn.PReLU(channels))

        self.max_pool_1 = nn.MaxPool2d(2, 2)
        self.avg_pool_1 = nn.AvgPool2d(2, 2)
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        self.avg_pool_2 = nn.AvgPool2d(2, 2)

        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, c, height, width = x.size()
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)

        pool_c1 = self.max_pool_1(c1) + self.avg_pool_1(c1)
        pool_c2 = self.max_pool_2(c2) + self.avg_pool_2(c2)

        pool_c1_reshape = pool_c1.view(pool_c1.size(0), pool_c1.size(1), -1)
        pool_c2_reshape = pool_c2.view(pool_c2.size(0), pool_c2.size(1), -1)

        matrix = torch.bmm(pool_c1_reshape, pool_c2_reshape.permute(0, 2, 1))
        attention = self.softmax(matrix)
        #print(attention.shape)

        refined = torch.bmm(attention, c3.view(c3.size(0), c3.size(1), -1))
        refined = refined.view(batchsize, c, height, width)

        out = self.gamma * refined + x
        return out

class LightChannelAttentionModule(nn.Module):
    def __init__(self, channels):
        super(LightChannelAttentionModule, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels),
                                    nn.PReLU(channels))
        self.conv_2 = nn.Sequential(nn.Conv2d(channels, channels//2, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels//2),
                                    nn.PReLU(channels//2))
        self.conv_3 = nn.Sequential(nn.Conv2d(channels, channels//2, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels//2),
                                    nn.PReLU(channels//2))

        self.max_pool_1 = nn.MaxPool2d(2, 2)
        self.avg_pool_1 = nn.AvgPool2d(2, 2)
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        self.avg_pool_2 = nn.AvgPool2d(2, 2)

        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, c, height, width = x.size()
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)

        pool_c1 = self.max_pool_1(c1) + self.avg_pool_1(c1)
        pool_c2 = self.max_pool_2(c2) + self.avg_pool_2(c2)

        pool_c1_reshape = pool_c1.view(pool_c1.size(0), pool_c1.size(1), -1)
        pool_c2_reshape = pool_c2.view(pool_c2.size(0), pool_c2.size(1), -1)
        c3_reshape = c3.view(c3.size(0), c3.size(1), -1)

        matrix = torch.bmm(pool_c1_reshape, pool_c2_reshape.permute(0, 2, 1))
        attention = self.softmax(matrix)
        #print(attention.shape)

        refined = torch.bmm(attention, c3_reshape)
        refined = refined.view(batchsize, c, height, width)

        out = self.gamma * refined + x
        return out

class NaiveSpatialAttentionModule(nn.Module):
    def __init__(self, channels):
        super(NaiveSpatialAttentionModule, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channels, channels//16, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels//16),
                                    nn.PReLU(channels//16))
        self.conv_2 = nn.Sequential(nn.Conv2d(channels, channels//16, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels//16),
                                    nn.PReLU(channels//16))
        self.conv_3 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels),
                                    nn.PReLU(channels))

        self.conv_4 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels))

        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, c, height, width = x.size()

        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)

        c1_reshape = c1.view(batchsize, -1, height * width)
        c2_reshape = c2.view(batchsize, -1, height * width)
        c3_reshape = c3.view(batchsize, -1, height * width)

        matrix = torch.bmm(c2_reshape.permute(0, 2, 1), c1_reshape)
        attention = self.softmax(matrix)

        refined = torch.bmm(c3_reshape, attention.permute(0, 2, 1))
        refined = refined.view(batchsize, c, height, width)

        out = self.gamma * refined + x
        return out
        #return self.conv_4(out)

class MaxPoolSpatialAttentionModule(nn.Module):
    def __init__(self, channels):
        super(MaxPoolSpatialAttentionModule, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channels, channels // 16, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels // 16),
                                    nn.PReLU(channels // 16))
        self.conv_2 = nn.Sequential(nn.Conv2d(channels, channels // 16, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels // 16),
                                    nn.PReLU(channels // 16))
        self.conv_3 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels),
                                    nn.PReLU(channels))

        self.conv_4 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels))

        self.max_pool_1 = nn.MaxPool2d(2, 2)
        self.max_pool_3 = nn.MaxPool2d(2, 2)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)

        c1_pool = self.max_pool_1(c1)
        c3_pool = self.max_pool_3(c3)

        c1_pool_reshape = c1_pool.view(c1_pool.size(0), c1_pool.size(1), -1)
        c3_pool_reshape = c3_pool.view(c3_pool.size(0), c3_pool.size(1), -1)

        c2_reshape = c2.view(c2.size(0), c2.size(1), -1)

        matrix = torch.bmm(c2_reshape.permute(0, 2, 1), c1_pool_reshape)
        attention = self.softmax(matrix)
        #print(attention.shape)

        refined = torch.bmm(c3_pool_reshape, attention.permute(0, 2, 1))
        refined = refined.view(x.size(0), x.size(1), x.size(2), x.size(3))

        out = self.gamma*refined + x
        return out
        #return self.conv_4(out)

class MaxPoolSpatialAttentionModule_2(nn.Module):
    # remove the weighted parameter gamma.
    def __init__(self, channels):
        super(MaxPoolSpatialAttentionModule_2, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channels, channels // 16, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels // 16),
                                    nn.PReLU(channels // 16))
        self.conv_2 = nn.Sequential(nn.Conv2d(channels, channels // 16, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels // 16),
                                    nn.PReLU(channels // 16))
        self.conv_3 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels),
                                    nn.PReLU(channels))

        self.conv_4 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels))

        self.max_pool_1 = nn.MaxPool2d(2, 2)
        self.max_pool_3 = nn.MaxPool2d(2, 2)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)

        c1_pool = self.max_pool_1(c1)
        c3_pool = self.max_pool_3(c3)

        c1_pool_reshape = c1_pool.view(c1_pool.size(0), c1_pool.size(1), -1)
        c3_pool_reshape = c3_pool.view(c3_pool.size(0), c3_pool.size(1), -1)

        c2_reshape = c2.view(c2.size(0), c2.size(1), -1)

        matrix = torch.bmm(c2_reshape.permute(0, 2, 1), c1_pool_reshape)
        attention = self.softmax(matrix)

        refined = torch.bmm(c3_pool_reshape, attention.permute(0, 2, 1))
        refined = refined.view(x.size(0), x.size(1), x.size(2), x.size(3))

        out = refined + x
        return out
        #return self.conv_4(out)

class LightSelfSpatialAttentionModule(nn.Module):
    def __init__(self, channels):
        super(LightSelfSpatialAttentionModule, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channels, channels // 16, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels // 16),
                                    nn.PReLU(channels // 16))
        self.conv_2 = nn.Sequential(nn.Conv2d(channels, channels // 16, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels // 16),
                                    nn.PReLU(channels // 16))
        self.conv_3 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels),
                                    nn.PReLU(channels))

        self.conv_4 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels))

        self.max_pool_1 = nn.MaxPool2d(2, 2)
        self.avg_pool_1 = nn.AvgPool2d(2, 2)
        self.max_pool_3 = nn.MaxPool2d(2, 2)
        self.avg_pool_3 = nn.AvgPool2d(2, 2)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)

        c1_pool = self.max_pool_1(c1) + self.avg_pool_1(c1)
        c3_pool = self.max_pool_3(c3) + self.avg_pool_3(c3)

        c1_pool_reshape = c1_pool.view(c1_pool.size(0), c1_pool.size(1), -1)
        c3_pool_reshape = c3_pool.view(c3_pool.size(0), c3_pool.size(1), -1)

        c2_reshape = c2.view(c2.size(0), c2.size(1), -1)

        matrix = torch.bmm(c2_reshape.permute(0, 2, 1), c1_pool_reshape)
        attention = self.softmax(matrix)

        refined = torch.bmm(c3_pool_reshape, attention.permute(0, 2, 1))
        refined = refined.view(x.size(0), x.size(1), x.size(2), x.size(3))

        out = self.gamma * refined + x
        return out


class BottleNeck_IR(nn.Module):
    '''Improved Residual Bottlenecks'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR_SSA(nn.Module):
    '''Improved Residual Bottlenecks with Self Spatial Attention Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_SSA, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       NaiveSpatialAttentionModule(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR_SSA_Tiny(nn.Module):
    '''Improved Residual Bottlenecks with Self Spatial Attention Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_SSA_Tiny, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       LightSelfSpatialAttentionModule(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR_SCA(nn.Module):
    '''Improved Residual Bottlenecks with Self Spatial Attention Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_SCA, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       NaiveChannelAttentionModule(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR_SCA_Tiny(nn.Module):
    '''Improved Residual Bottlenecks with Self Spatial Attention Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_SCA_Tiny, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       LightChannelAttentionModule(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res


class BottleNeck_IR_SRAM(nn.Module):
    '''Improved Residual Bottleneck with Self Channel Attention and Self Spatial Attention'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_SRAM, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       NaiveSpatialAttentionModule(out_channel),
                                       LightSelfSpatialAttentionModule(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res


filter_list = [64, 64, 128, 256, 512]
def get_layers(num_layers):
    if num_layers == 50:
        return [3, 4, 14, 3]
    elif num_layers == 100:
        return [3, 13, 30, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]

class SRAMResNet_IR(nn.Module):
    def __init__(self, num_layers, feature_dim=512, mode='ir', drop_ratio=0.4, filter_list=filter_list):
        super(SRAMResNet_IR, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'ir_sca', 'ir_sca_tiny','ir_ssa', 'ir_ssa_tiny', 'ir_sram', 'ir_pam', 'ir_cam'], 'mode should be ir, ir_sca, ir_sca_tiny, ir_ssa, ir_ssa_tiny, ir_sram'
        layers = get_layers(num_layers)
        if mode == 'ir':
            block = BottleNeck_IR
        elif mode == 'ir_ssa':
            block = BottleNeck_IR_SSA
        elif mode == 'ir_ssa_tiny':
            block = BottleNeck_IR_SSA_Tiny
        elif mode == 'ir_sca':
            block = BottleNeck_IR_SCA
        elif mode == 'ir_sca_tiny':
            block = BottleNeck_IR_SCA_Tiny
        elif mode == 'ir_sram':
            block = BottleNeck_IR_SRAM

        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)

        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)

        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)

        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 7 * 7, feature_dim),
                                          nn.BatchNorm1d(feature_dim))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride, False))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)

        return x


if __name__ == "__main__":
    input = torch.Tensor(2, 16, 112, 112)
    #net = SRAMResNet_IR(50, mode='ir_ssa')
    #net = NaiveSpatialAttentionModule(16)
    #net = MaxPoolSpatialAttentionModule_2(16)
    #net = LightSelfSpatialAttentionModule(16)
    net = NaiveChannelAttentionModule(16)
    #net = LightChannelAttentionModule(16)


    out = net(input)
    print(out.shape)

    '''
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    input, net = input.to(device), net.to(device)

    # calculate the inference time:
    net.eval()
    with torch.no_grad():
        total = 0
        for i in range(100):
            t1 = time.time()
            x = net(input)
            t2 = time.time()
            print('current_time: %04d: '%i, t2-t1)
            if i > 0:
                total += (t2-t1)
        print('total time: ', total, ' average time: ', total/99)
    x = net(input)
    print(x.shape)

    # calculate the inference time:
    '''