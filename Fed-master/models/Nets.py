#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class mnist_model2(nn.Module):
    def __init__(self, args):
        super(mnist_model2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
        )
        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # 输出相当于16个通过5*5卷积核得到的特征图   https://blog.csdn.net/Aiden_yan/article/details/122570731
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 将3通道展平成1通道
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

''' 
这是一个在 CIFAR-10 数据集上的一个常见的 CNN 网络架构（LeNet-like），在这个网络中全连接层的第一层的输入节点数 5516 是这样计算得到的：
输入为 3 通道的 32 * 32 图像 （假设 CIFAR-10）
第一卷积层 conv1 是一个带有 6 个过滤器的卷积，卷积核的大小为 5，没有设置 padding，因此在经过 conv1 层后，图像大小变为了 28 * 28 （因为 (32 - 5) / 1 + 1 = 28），并且通道数变为了 6。
然后经过了 pool 层，这个层的池化窗口和步长都是 2，因此在这个层后，图像的大小变为了 14 * 14 （因为 28 / 2 = 14），通道数仍然是 6。
第二卷积层 conv2 是一个带有 16 个过滤器的卷积，卷积核的大小为 5，没有设置 padding，因此在经过 conv2 层后，图像大小变为了 10 * 10 （因为 (14 - 5) / 1 + 1 = 10），并且通道数变为了16。
再次经过 pool 层，图像的大小变为了 5 * 5 （因为 10 / 2 = 5），通道数仍然是16。
接着，全连接层 fc1 接收的输入将是 pool 层输出的展平版本，也就是说它将接收一个长度为 5 * 5 * 16 = 400 的一维向量，这就是为什么全连接层 fc1 的输入特征数为 5 * 5 * 16。
所以，全连接层的第一层的输入特征数 5516，就是根据经过了两次卷积和两次最大池化后的输出特征映射的尺寸来计算得出的
'''

'''
在卷积神经网络中，填充（Padding）是在卷积层中的一种常见处理方式，主要有以下几个作用：

保持空间尺寸：在进行卷积操作时，由于卷积核需要在输入图像上滑动，如果没有使用填充，那么输出的特征图尺寸（即卷积的结果）会比输入图像尺寸小。这种现象在深层的卷积神经网络中尤为显著，可能会导致空间尺寸缩小得过快。通过适当添加填充，我们可以对此进行控制，使得卷积操作后的输出特征映射尺寸保持不变，或者至少缩小得较慢。

保持更多的边边角角信息：在没有填充的情况下，卷积核主要操作于输入图像的中心区域，而对于边缘部分的覆盖较少。通过添加填充，我们可以让卷积核有更多的机会处理边缘部分的信息，这样可以帮助网络更好地学习边缘部分的特征。

添加非线性：通过在卷积操作中加入填充，并与非线性激活函数（如ReLU）结合使用，可以在每一层中添加更多的非线性，加强模型的表达能力。
'''




# 如果全连接层的节点数太少，则可能导致模型复杂度不够，无法学习到数据的所有特征；如果节点数设置过多，可能会导致模型过拟合。
# 此外，还需要注意的是：全连接层的输入和输出节点数对计算资源和模型效果都有影响，所以在选择节点数时，一般需要综合考虑模型效果、过拟合风险，以及计算资源的限制。
class CNNCifar_New(nn.Module):
    def __init__(self, args):
        super(CNNCifar_New, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias= True)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 将3通道展平成1通道
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class cnncifar_new2(nn.Module):
    def __init__(self, args):
        super(cnncifar_new2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
        )
        self.linear = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class cnncifar_new2_moon(nn.Module):
    def __init__(self, args):
        super(cnncifar_new2_moon, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
        )
        self.linear = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        y = self.linear(x)
        return x, y