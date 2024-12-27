#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from numpy import sign,ravel
import numpy as np


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# 这里的index即之前的 ch-client 的字典  management
def FedAvg_index_client(w, index, ch_id):
    w_avg_i = copy.deepcopy(w[index[ch_id][0]])
    for k in w_avg_i.keys():
        for i in range(1, len(index[ch_id])):
            w_avg_i[k] += w[index[ch_id][i]][k]
        w_avg_i[k] = torch.div(w_avg_i[k], len(index[ch_id]))
    # ch 和 其子用户聚合 , 不是数据占比1/2  可以改成数据占比  1：10
    for k in w_avg_i.keys():
        w_avg_i[k] += w[ch_id][k]
        w_avg_i[k] = torch.div(w_avg_i[k], 2)
    return w_avg_i



# 这里仅做简单的判断，根据其client_id判断是否是大数据用户，在实际的情况中应该通过 data量和emd距离进行综合考虑来判断
# 0<=client_id<=19 仅对0-19的用户进行特殊处理
# id 中包含本轮参与fl的用户
# bug 确保第一个用户不是0-19的用户，其次跳过第一个用户的累加； 探讨仅聚合卷积层不聚合全连接层 会不会致使两者不匹配导致模型性能很差
def Fed_personal(w, id):
    for x in id:
        if x > 19:
            w_avg = copy.deepcopy(w[x])
            get_id = x
            break
    skip_keys = ['fc3.weight', 'fc3.bias']
    all_agg = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
    new_id = [i for i in id if i > 19]
    for k in all_agg:
        # 求和
        for i in id:
            if i != get_id:  # 不对载入用户做重复计算
                w_avg[k] += w[i][k]
        # 平均
        w_avg[k] = torch.div(w_avg[k], len(id))

    for k in skip_keys:
        for i in new_id:
            if i != get_id:
                w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(new_id))
    return w_avg

def FedAvgWithCmfl(w,w_old,threshold=0.7,mute=True,checking=None):
    w_delta = DeltaWeights_cmfl(w,w_old)
    w_davg = FedAvg(w_delta)
    agreeThres = 0
    for k in w_davg.keys():
        templist = w_davg[k].cpu().numpy().reshape(-1)
        agreeThres += len(templist)
    agreeThres *= threshold
    w_agree = []

    maxagree = 0
    maxindex = 0

    checklist = [False] * len(w_delta)
    
    for i in range(len(w_delta)):
        agreeCount = 0
        for k in w_davg.keys():
            templist1 = w_davg[k].cpu().numpy().reshape(-1)
            templist2 = w_delta[i][k].cpu().numpy().reshape(-1)
            for j in range(len(templist1)):
                if sign(templist1[j]) == sign(templist2[j]):
                    agreeCount += 1
        if agreeCount >= agreeThres:
            w_agree.append(w[i])
            checklist[i] = True
        if maxagree < agreeCount:
            maxagree = agreeCount
            maxindex = i
    if len(w_agree) > 0:
        w_avg = FedAvg(w_agree)
    else:
        w_avg = w[maxindex]

    if checking is not None:
        clientId = checking[0].copy()
        iidThreshold = checking[1]
        rec = [0] * 4
        for i in range(len(clientId)):
            check = 0
            if clientId[i] >= iidThreshold: # nIID
                check += 2
            if checklist[i] is False: #killed
                check += 1
            rec[check] += 1
        print("Cutting : IID (%2d/%2d), non (%2d/%2d)" %tuple(rec))
        

    if mute == False and checking is None:
        print("CMFL: %d out of %d is accepted" % (len(w_agree),len(w)))
    return w_avg

    
def DeltaWeights(w, w_old):
    deltas = copy.deepcopy(w)
    for k in w[0].keys():
        for i in range(len(w)):
            # reshape(-1)将其扁平化方便做差
            deltas[i][k] = deltas[i][k].cpu().numpy().reshape(-1)
            deltas[i][k] -= w_old[k].cpu().numpy().reshape(-1)
    return deltas

def DeltaWeights_cmfl(w,w_old):
    deltas = copy.deepcopy(w)
    for k in w[0].keys():
        for i in range(len(w)):
            deltas[i][k] -= w_old[k]
    return deltas