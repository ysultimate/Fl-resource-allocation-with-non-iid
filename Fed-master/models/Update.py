#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import copy
from torch.nn.functional import cosine_similarity
import numpy as np
import random
# from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, locallr=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lr = locallr
        if locallr is None:
            self.l = self.args.lr

    def train(self, net, flag):
        net.train()
        # train and update， 分为学习率衰减和不衰减，weight_decay项是用来防止过拟合的，正则项前面的系数
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=5e-4)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                if flag == 2:
                    _, log_probs = net(images)
                else:
                    log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

            # 本地更新时每个epoch之间都需要手动进行学习率衰减
            self.lr = self.lr * self.args.lrd
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.lr


class LocalUpdate_moon(object):
    def __init__(self, args, dataset=None, idxs=None, locallr=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lr = locallr
        if locallr is None:
            self.l = self.args.lr

    def train(self, net, previous_nets):
        global_ = copy.deepcopy(net).to(torch.device('cuda:0'))
        previous_nets_ = copy.deepcopy(net).to(torch.device('cuda:0'))
        previous_nets_.load_state_dict(previous_nets)
        net.train()
        # train and update， 分为学习率衰减和不衰减，weight_decay项是用来防止过拟合的，正则项前面的系数
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=5e-4)
        epoch_loss = []

        # sim 值
        cnt = 0
        cos = torch.nn.CosineSimilarity(dim=-1)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                if iter == 0:
                    _, y_pre = net(images)
                    loss = self.loss_func(y_pre, labels)
                else:
                    images.requires_grad = False
                    labels.requires_grad = False
                    labels = labels.long()

                    pro1, out = net(images)
                    pro2, _ = global_(images)
                    posi = cos(pro1, pro2)
                    logits = posi.reshape(-1, 1)

                    pro3, _ = previous_nets_(images)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                    logits /= 0.5
                    labels2 = torch.zeros(images.size(0)).cuda().long()
                    loss2 = 4 * self.loss_func(logits, labels2)
                    loss1 = self.loss_func(out, labels)
                    loss = loss1 + loss2
                    # z_curr = net.get_last_features(images, detach=False)
                    # z_global = global_.get_last_features(images, detach=True)
                    # z_prev = previous_nets_.get_last_features(images, detach=True)
                    # logits = net.classifier(z_curr)
                    # loss_sup = self.loss_func(logits, labels)
                    # loss_con = -torch.log(
                    #     torch.exp(
                    #         cosine_similarity(z_curr.flatten(1), z_global.flatten(1))
                    #         / 0.5
                    #     )
                    #     / (
                    #             torch.exp(
                    #                 cosine_similarity(z_prev.flatten(1), z_curr.flatten(1))
                    #                 / 0.5
                    #             )
                    #             + torch.exp(
                    #         cosine_similarity(z_curr.flatten(1), z_global.flatten(1))
                    #         / 0.5
                    #     )
                    #     )
                    # )
                    # loss = loss_sup + 5 * torch.mean(loss_con)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

            # 本地更新时每个epoch之间都需要手动进行学习率衰减
            self.lr = self.lr * self.args.lrd
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.lr

class LocalUpdate_prox(object):
    def __init__(self, args, dataset=None, idxs=None, locallr=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lr = locallr
        if locallr is None:
            self.l = self.args.lr

    def train(self, net):
        global_ = copy.deepcopy(net).to(torch.device('cuda:0'))
        global_weight_collector = list(global_.cuda().parameters())
        net.train()
        # train and update， 分为学习率衰减和不衰减，weight_decay项是用来防止过拟合的，正则项前面的系数
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=5e-4)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.long()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                # for fedprox
                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((0.5 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

            # 本地更新时每个epoch之间都需要手动进行学习率衰减
            self.lr = self.lr * self.args.lrd
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.lr


class Serial_LocalUpdate(object):
    # idxs 是 某用户对应的样本索引
    def __init__(self, args, dataset=None, idxs=None, locallr=None):
        self.args = args
        self.datasets = dataset
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dict_client_sample = idxs
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lr = locallr
        if locallr is None:
            self.l = self.args.lr

    def train(self, net, client_id):
        net.train()
        # train and update， 分为学习率衰减和不衰减，weight_decay项是用来防止过拟合的，正则项前面的系数
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=5e-4)
        lr_initial = self.lr
        epoch_loss = { i : [] for i in client_id }
        for idx in client_id:
            # 学习率初始化，防止串行联邦学习率下降过快
            self.lr = lr_initial
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
            train_data = DataLoader(DatasetSplit(self.datasets, self.dict_client_sample[idx]), batch_size=100,
                                    shuffle=True)
            # 子联邦中的串行迭代过程
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(train_data):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                   100. * batch_idx / len(self.ldr_train), loss.item()))
                    batch_loss.append(loss.item())

                # 本地更新时每个epoch之间都需要手动进行学习率衰减
                self.lr = self.lr * self.args.lrd
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr

                epoch_loss[idx].append(sum(batch_loss)/len(batch_loss))
            epoch_loss[idx] = sum(epoch_loss[idx]) / len(epoch_loss[idx])
        return net.state_dict(), sum(epoch_loss.values()) / len(epoch_loss), self.lr

# 针对某一个用户的singlebgd，直接将该用户的所有数据打包成一个batch进行fl
class SingleBgdUpdate(object):
    def __init__(self, args, data_num=None, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.data_num = data_num
        if data_num is None:
            self.data_num = len(idxs)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.data_num, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0)
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
        # 如果不是轻度分析就传递回整个模型的权重字典
        if self.args.light_analytics is False:
            gradients = {}
            for name, parameter in net.named_parameters():
                gradients[name] = parameter.grad.clone()
            return net.state_dict(), gradients
        # 这里是轻度分析+多样性分析
        lightDict = {}
        gradients = {}
        # 仅传回全连接最后一层的权重，因为根据emd分析，在进行non-iid训练时全连接最后一层显示出的散度最大
        lightDict['fc3.weight'] = net.state_dict()['fc3.weight']
        lightDict['fc3.bias'] = net.state_dict()['fc3.bias']
        gradients['fc3.weight'] = net.fc3.weight.grad.clone()
        # gradients['fc3.bias'] = net.fc3.bias.grad.clone()
        return lightDict, gradients


class MiniFL_SingleBgdUpdate(object):
    def __init__(self, args, data_num=None, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dict_client_sample = idxs
        self.merged_list = [item for sublist in self.dict_client_sample.values() for item in sublist]
        self.data_num = data_num
        if data_num is None:
            self.data_num = len(self.merged_list)
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.merged_list), batch_size=self.data_num, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0)
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
        # 如果不是轻度分析就传递回整个模型的权重字典
        if self.args.light_analytics is False:
            return net.state_dict()
        # 这里是轻度分析
        lightDict = {}
        # 仅传回全连接最后一层的权重，因为根据emd分析，在进行non-iid训练时全连接最后一层显示出的散度最大
        lightDict['fc3.weight'] = net.state_dict()['fc3.weight']
        lightDict['fc3.bias'] = net.state_dict()['fc3.bias']
        return lightDict


def simulate_element_loss_and_recovery(original_dict, new_dict, drop_rate, device='cpu'):
    recovered_state_dict = {}
    for key in new_dict.keys():
        new_tensor = new_dict[key].clone().to(device)  # 移动到指定设备
        original_tensor = original_dict[key].to(device)  # 移动到指定设备
        mask = torch.rand(new_tensor.size(), device=device) > drop_rate  # 在指定设备上生成掩码

        # 用于调试：打印 mask
        # print(f"Mask for {key}: {mask.cpu().numpy()}")

        # 用original_tensor的值替换被mask掉的元素
        new_tensor = torch.where(mask, new_tensor, original_tensor)

        recovered_state_dict[key] = new_tensor
    return recovered_state_dict
