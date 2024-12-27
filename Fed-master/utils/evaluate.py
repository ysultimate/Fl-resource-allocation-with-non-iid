import copy
import torch
from torch import nn
from numpy import sign,ravel
import numpy as np
import math
from itertools import combinations

import models.Fed as Fed


def deltaWeightEvaluate(w,w_old):
    deltas = copy.deepcopy(w)
    for k in w.keys():
        deltas[k] -= w_old[k]
    weights = None
    for k in deltas.keys():
        newlist = deltas[k].numpy().reshape(-1)
        if weights is None:
            weights = newlist
        else:
            weights = np.append(weights,newlist)
    print('Weights size: %d' % len(weights))

    wmax = weights.max()
    wmin = weights.min()
    wavg = weights.mean()
    wstd = weights.std()

    weights = np.sort(weights)
    q1 = weights[int(len(weights)/4)]
    q3 = weights[int(len(weights)*3/4)]

    print("max: %.4f, min: %.4f, avg: %.4f, std: %.4f" % (wmax,wmin,wavg,wstd))
    print("range: %.4f, IQR: %.4f, IQR/range: %.4f, std/range:  %.4f" % (wmax-wmin , q3-q1 , (q3-q1)/(wmax-wmin) , wstd/(wmax-wmin)))

# w是list里面存放有许多本路参与的模型
# w_old是用来求delta
def l2NormEvaluate(w_old, w, replace_avg=None):
    if w_old is None: # 1st iter
        w_delta = w
    else:
        # 得到全局模型和本次迭代出的各个本地模型之间的delta值
        w_delta = Fed.DeltaWeights(w, w_old)

    # 将delta 进行平均
    if replace_avg is None:
        w_davg = copy.deepcopy(w_delta[0])
        for k in w_davg.keys():
            for i in range(1, len(w_delta)):
                w_davg[k] += w_delta[i][k]
            w_davg[k] = w_davg[k] / len(w_delta)
    else:
        ravglist = [replace_avg]
        w_davgl = Fed.DeltaWeights(ravglist,w_old)
        w_davg = w_davgl[0]

    # l2norms的长度为本轮参与的client数
    l2norms = [0] * len(w)
    for k in w_davg.keys():
        # avlist获取了平均后的delta权重某一层的参数
        avlist = w_davg[k]
        for i in range(len(w)):
            # ilist是client-i 和 本轮下发的全局模型之间的delta值，这里就是求： Δw_avg-Δw_i（轻量级fl分析就是仅分析最后一层全连接层）
            ilist = w_delta[i][k]
            diff = avlist - ilist
            # 这里的求和是各个client根据各个模型层之间的l2值求和，⭐难道不需要先开平方后再求和再开平方吗
            l2norms[i] += np.linalg.norm(diff,ord=2)

    return l2norms

def Similarity_analysis(grad, num_this_round):
    flat_grad = []
    mean =( (num_this_round-1)*num_this_round )/ 2
    sum_similarity = 0
    # 轻度分析，仅提取某一层（通过修改“fc3.weight”来决定你想选的层）
    # print(grad)
    for i in range(len(grad)):
        flat = grad[i]['fc3.weight']
        flat = flat.view(-1)
        flat_grad.append(flat)
    for pair in combinations(flat_grad,2):
        # 这里的*pair 将pair中的两个参数直接传入cosine函数当中
        sum_similarity = sum_similarity + cosine(*pair)
    # sum_similarity = sum_similarity.item()
    sum_similarity = sum_similarity / mean
    return  sum_similarity

# 用于计算两者的余弦相似1度，区间转化至 （0-1）     0表示最不相似
def cosine(a,b):
    cos_sim = torch.cosine_similarity(a, b, dim=0)
    cos_sim = cos_sim.item()
    trans_cos = (cos_sim + 1) / 2
    return  trans_cos

# fa进行联邦分析，通过faf这个超参数来调试，目前faf 都是取>1的值，即前200轮会持续进行下去
def FA_round(args,iter):
    if args.faf < 0:
        return False

    if args.faf == 0:
        return True
    
    if args.faf == 1:
        if iter < 200:
            return True
    return False