#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# import bridson
import random
import numpy as np
from torchvision import datasets, transforms
import math
# mnist的训练集有50000个样本
def mnist_iid(dataset, num_users=200,num_samples=300):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    data_num = len(dataset)
    class_num = 10
    # 给各个sample上索引
    idxs = np.arange(data_num)
    # 提取训练集中的样本标签，每个元素都是属于0-9的值
    labels = dataset.train_labels.numpy()
    dict_users = {}

    overalldist = [0] * class_num
    # 统计各个class有多少个样本
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    # 在位置0处（开头）插入元素0
    overalldist.insert(0,0)
    # 逐项向后累加，每个overalldist[i]中存放的是前面的class累加至自己的sample量
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    # 删除heads[10]中的元素，sample总数
    del[heads[class_num]]

    # sort labels ， 形成一个2维数组的形式，上面是索引，下面是标签
    idxs_labels = np.vstack((idxs, labels))
    # 将idxs_labels按class类划分
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # 取划分好了的sample标签
    idxs = idxs_labels[0,:]

    dominances = []
    # 初始化count字典
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,dominance=0,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances


def cifar_iid(dataset, num_users=200, num_samples=250):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    data_num = len(dataset)
    class_num = 10
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,dominance=0,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances
# idxs 是按label从小到大分类好的 各个sample的索引号
# result = add_data(result,idxs,heads,overalldist,i,classdist[i],counts)
# group = i 表示第几类，overalldist表示各个类有多少个样本（累计的）
# 这个函数是用来给某个client分配各个类的数据样本的函数
def add_data(data,idxs,head,overall,group,num,counts):
    # head比overall少最后一项
    # remaining 表示 class i 此时还剩多少个sample能用
    # num 表示该class应该划分多少数据
    remaining = overall[group+1] - head[group]
    # counts是记录各个class分配数据的字典？ 初始化为 i ：0
    counts[group] += num
    # 如果剩余的数据量满足 需要分配的数据量
    if remaining > num:
        # 级联函数， 从idxs中取 对应class起点开始--num的数据出来
        data = np.concatenate((data,idxs[head[group] : head[group]+num]))
        # 将head头指针向后移动
        head[group] += num

        return data

    # 如果剩余数据不满足待分配的数据量，先将剩余的数据分配完
    data = np.concatenate((data,idxs[head[group] : head[group]+remaining]))
    # 剩余数据分配完后将head头指针重新指回起点
    head[group] = overall[group]

    # shuffle order of that group
    temp = idxs[overall[group] : overall[group+1]]
    # 每分配完一类数据之后，将该类数据在idxs中的索引洗牌
    random.shuffle(temp)
    idxs[overall[group] : overall[group+1]] = temp

    # 计算还有多少数据还亟待分配
    filling = num - remaining
    data = np.concatenate((data,idxs[head[group] : head[group]+filling]))
    # head指针后移                
    head[group] += filling
    return data


# 自己的实验分成4类client（20 20 80 80）
# 12.11 继续测试 缩小超级用户的比例进行fl（10 10 90 90）
# 6.9  调成只有1类
def myfourclass_client(dataset, client_num, heads, overalldist, idxs, counts, client_id, dir, classNum, dominance=None):
    # 分成4层按照用户的索引0-24，25-49，50-74，75-99来划分ABCD四组
    # 改成2类
    sample_num = 500
    if 0 <= client_id <= (int(client_num*0.2)-1):
        sample_num = 1000

    elif int(client_num*0.2) <= client_id <= (client_num-1):
        # sample_num = random.randint(100, 200)
        sample_num = 100

    # elif 20<=client_id<=109:
    #     sample_num = 100
    #     dominance = float(random.uniform(0.5, 3))
    # else:
    #     sample_num = 100
    #     dominance = float(random.uniform(0.008, 0.012))

    classdist = [0,0,0,0,0,0,0,0,0,0]
    classdist[random.randint(0,9)] =1
    EMD = sum(abs(num - 1/classNum) for num in classdist)
    for i in range(len(classdist)):
            # 生成各个class类别应该划分多少个样本
        classdist[i] = int(classdist[i] * sample_num)
        # 检查数据是否够，可能由于取整问题不足
    samplesleft = sample_num - sum(classdist)
        # 随机生成0-（classnum-1）中的一个整数
    drawclass = int(np.random.randint(classNum))
        # 将数据补足，现在classdist中存放的是各个标签的数据在这个用户中应该取多少个
    classdist[drawclass] += samplesleft

    # result 表示该client应该划分的data的sample索引列表
    result = np.array([], dtype='int64')
    # 在循环中处理这个client的 n（一般是10个） 个类的数据
    for i in range(classNum):
        result = add_data(result, idxs, heads, overalldist, i, classdist[i], counts)
    return result, dominance , sample_num, EMD, classdist


# num_sample是各个用户需要分多少数据
def myfourclass_lab(flag_dataset, dataset, num_users, class_num, dir):
    data_num = len(dataset)
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0, 0)
    for i in range(1, class_num + 1):
        overalldist[i] += overalldist[i - 1]
    heads = overalldist.copy()
    del [heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    dominances = []
    client_sample = []
    emd = []
    counts = {}
    # 用于存放各用户的各个样本拥有的数量
    overall_client_data_population = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi ,client_sample_num, emd_n, client_data_population = myfourclass_client(flag_dataset, num_users, heads, overalldist, idxs, counts, i, dir,
                                                     classNum=class_num)
        dict_users[i] = subset
        overall_client_data_population[i] = client_data_population
        dominances.append(domi)
        client_sample.append(client_sample_num)
        emd.append(emd_n)
    # 计算总共数据量
    total_sample = sum(client_sample)
    # 计算价值中的数据量价值
    datanum_value = [(x / total_sample) for x in client_sample]
    datanum_value = [x*100 for x in datanum_value]
    return dict_users, dominances, client_sample, emd, datanum_value, overall_client_data_population

# iid传进来的dominance 是 0
def dominance_client(heads,overalldist,idxs,counts,sampleNum,classNum,dominance=None,dClass=None):
    if dominance is None:
        dominance = random.uniform(0,1.0)
    if dClass is None:
        sortcounts = sorted(counts.items(),key=lambda x:x[1],reverse=False)
        dClass = sortcounts[0][0]

    dominance = float(dominance)
    # math.floor是向下取整
    iidClassSize = math.floor(sampleNum *(1 - dominance) / classNum)
    # iid情况下 nonclasssize=0
    nonClassSize = sampleNum - classNum * iidClassSize
    result = np.array([], dtype='int64')
    result = add_data(result,idxs,heads,overalldist,dClass,nonClassSize,counts)

    for i in range(classNum):
        result = add_data(result,idxs,heads,overalldist,i,iidClassSize,counts)
    
    return result, dominance

def complex_skewness_mnist(dataset, num_users, num_samples):
    data_num = len(dataset)
    class_num = 10
    idxs = np.arange(data_num)
    labels = dataset.train_labels.numpy()
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances

def uni_skewness_cifar(dataset, num_users, num_samples, class_num=10):
    data_num = len(dataset)
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances

# 这里没改之前是每个client随机分配几个class的数据
# 用int(client_num / 10) 来确定各个client应该取哪个类别的数据即可
def fewclass_dominance_client(heads,overalldist,idxs,counts,client_id,sampleNum,classNum,dominance=None,dClass=None):
    localClass = int(client_id / 10)
    if dominance is None:
        dominance = random.uniform(0,1.0)
    dominance = float(dominance)
    datanum = 500
    result = np.array([], dtype='int64')
    result = add_data(result,idxs,heads,overalldist,localClass,datanum,counts)
    return result, dominance
    # localClassNum = random.randint(1,10)
    # localClass = np.random.choice(10,size=localClassNum,replace=False)
    # if dominance is None:
    #     dominance = random.uniform(0,1.0)
    # if dClass is None:
    #     sortcounts = sorted(counts.items(),key=lambda x:x[1],reverse=False)
    #     for i in range(classNum):
    #         dClassCandidate = sortcounts[i][0]
    #         if dClassCandidate in localClass:
    #             dClass = dClassCandidate
    #             break
    #
    # dominance = float(dominance)
    #
    # iidClassSize = math.floor(sampleNum *(1 - dominance) / localClassNum)
    # nonClassSize = sampleNum - localClassNum * iidClassSize
    # result = np.array([], dtype='int64')
    # result = add_data(result,idxs,heads,overalldist,dClass,nonClassSize,counts)
    #
    # for i in localClass:
    #     result = add_data(result,idxs,heads,overalldist,i,iidClassSize,counts)
    #
    # return result, dominance - localClassNum

def fewclass_uni_skewness_cifar(dataset, num_users, num_samples, class_num=10):
    data_num = len(dataset)
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        subset, domi = fewclass_dominance_client(heads,overalldist,idxs,counts,i,sampleNum=num_samples,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances


def staged_skewness_cifar(dataset, num_users, num_samples, class_num=10):
    data_num = len(dataset)
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        if i < 10:
            subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num,dominance=0.0)
        elif i < 20:
            subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num,dominance=0.2)
        elif i < 30:
            subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num,dominance=0.4) 
        elif i < 40:
            subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num,dominance=0.6) 
        elif i < 50:
            subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num,dominance=0.8) 
        elif i < 60:
            subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num,dominance=1.0)   
        else:      
            subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances


def inversepareto_skewness_cifar(dataset, num_users, num_samples, class_num=10):
    data_num = len(dataset)
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        sample = np.random.pareto(2)
        while sample > 1 or sample < 0:
            sample = np.random.pareto(2)
        dominance = 1 - sample
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,dominance=dominance,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances


def pareto_skewness_cifar(dataset, num_users, num_samples, class_num=10):
    data_num = len(dataset)
    idxs = np.arange(data_num)
    labels = np.array(dataset.targets)
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in range(num_users):
        sample = np.random.pareto(2)
        while sample > 1 or sample < 0:
            sample = np.random.pareto(2)
        dominance = sample
        subset, domi = dominance_client(heads,overalldist,idxs,counts,sampleNum=num_samples,dominance=dominance,classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    
    return dict_users, dominances


def dirichlet_client(heads, overalldist, idxs, counts, sampleNum, classNum, alpha):
    alphatemp = [alpha] * classNum
    # 这里和我在fedavg代码中生成的方式不一样，这里仅生成1行，每一列代表一个class类别的比例
    # 这里的代码每个client生成的分布都是不同且是根据随机alpha值产生的
    classdist = list(np.random.dirichlet(alpha=alphatemp))
    for i in range(len(classdist)):
        # 生成各个class类别应该划分多少个样本
        classdist[i] = int(classdist[i] * sampleNum)
    # 检查数据是否够，可能由于取整问题不足
    samplesleft = sampleNum - sum(classdist)
    # 随机生成0-（classnum-1）中的一个整数
    drawclass = int(np.random.randint(classNum))
    # 将数据补足，现在classdist中存放的是各个标签的数据在这个用户中应该取多少个
    classdist[drawclass] += samplesleft

    # result 表示该client应该划分的data的sample索引列表
    result = np.array([], dtype='int64')
    # 在循环中处理这个client的 n（一般是10个） 个类的数据
    for i in range(classNum):
        result = add_data(result,idxs,heads,overalldist,i,classdist[i],counts)
    return result


# dataset传进来的是训练集，num_samples是预定要给各个client划分的数据量
def dirichlet_skewness_cifar(dataset, num_users, num_samples, class_num=10):
    data_num = len(dataset)
    idxs = np.arange(data_num)
    # 这里的labels np数组是仅含标签的，其对应的下标表示sample对应的索引号
    labels = np.array(dataset.targets)
    dict_users = {}

    # 统计各个类有多少个样本
    overalldist = [0] * class_num
    # len(labels) 是总数据量
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    # 在0号位置插入0
    overalldist.insert(0,0)
    for i in range(1,class_num+1):
        # 进行累加
        overalldist[i] += overalldist[i-1]
    heads = overalldist.copy()
    # 删除最后一项
    del[heads[class_num]]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # 将idxs按label从小到大组织起来
    idxs = idxs_labels[0,:]

    counts = {}
    for i in range(class_num):
        counts[i] = 0

    dominance = []

    # 在循环里处理各个client的数据
    for i in range(num_users):
        # 这里的domi值是dirichlet中数据的不平衡程度，α越小其离散程度越大；这里采用的是分层的dirichlet
        # if i % 2 != 0:
        #     domi = random.uniform(0.00001,0.2)
        # else:
        #     domi = random.uniform(0.2,3.0)
        # 这里采用非分层的dirichlet采样（需手动调整dir中的α值）
        domi = 0.15
        subset = dirichlet_client(heads,overalldist,idxs,counts,sampleNum=num_samples,classNum=class_num,alpha=domi)
        # 将各个client的数据添加至总字典中
        dict_users[i] = subset
        # 记录各个client的数据non-iid程度
        dominance.append(domi)

    return dict_users, dominance


def spell_partition(partition, labels,dominence=None):

    for nodeid, dataid in partition.items():
        record = [0] * 10
        for data in dataid:
            label = labels[data]
            record[label] += 1
        
        print("Client %d"% nodeid)
        if dominence is not None:
            print("Dominence: %.2f"% dominence[nodeid])

        for classid in range(len(record)):
            print("Class %d: %d" %(classid,record[classid]))
        print("\n\n")


def spell_data_usage(partition,datanum):
    record = [0]*datanum
    for dataid in partition.values():
        for data in dataid:
            record[data] += 1
    print("Avg usage: %.2f" % np.mean(record))
    print("Max usage: %d" % max(record))
    print("Min usage: %d" % min(record))


def coordinate_generation(radius, num_users):
    np.random.seed(24)
    # 生成随机角度
    thetas = 2 * np.pi * np.random.uniform(size=num_users)
    # 生成随机半径
    rs = np.sqrt(np.random.uniform(low=55 ** 2, high=(radius - 20) ** 2, size=num_users))
    # 把极坐标系下的点转换成笛卡尔坐标系（直角坐标系）
    coords = [(r * np.cos(theta), r * np.sin(theta)) for r, theta in zip(rs, thetas)]
    # 计算到圆心的距离 Km
    distances = [None for i in range(num_users)]
    for index, tup in enumerate(coords):
        distances[index] = np.sqrt(tup[0] ** 2 + tup[1] ** 2) / 1000
    return coords, distances


def generate_grid_points_in_annulus(inner_radius, outer_radius, num_users):
    # 计算每个点应占的面积
    total_area = np.pi * (outer_radius**2 - inner_radius**2)
    area_per_point = total_area / num_users
    side_length = np.sqrt(area_per_point)

    # 确定生成网格的大小
    num_per_side = int(outer_radius // side_length) + 1
    coords = []

    for i in range(-num_per_side, num_per_side + 1):
        for j in range(-num_per_side, num_per_side + 1):
            x, y = i * side_length, j * side_length
            distance = np.sqrt(x**2 + y**2)  # 计算点到原点的距离
            if inner_radius <= distance <= outer_radius:
                # 添加随机扰动
                x += np.random.uniform(-side_length / 2, side_length / 2)
                y += np.random.uniform(-side_length / 2, side_length / 2)
                coords.append((x, y))
    distances = [None for i in range(num_users)]
    for index, tup in enumerate(coords):
        distances[index] = np.sqrt(tup[0] ** 2 + tup[1] ** 2) / 1000
    return coords, distances


# def generate_poisson_disc_samples(min_dist, width, height, radius_inner, radius_outer, num_users):
#     scale_factor = 1  # 初始化比例因子
#     while True:
#         points = bridson.poisson_disc_samples(width=width * scale_factor, height=height * scale_factor,
#                                               r=min_dist * scale_factor)
#         # 过滤和转换坐标
#         coords = [(x - width / 2 * scale_factor, y - height / 2 * scale_factor) for x, y in points if
#                            radius_inner ** 2 <= ((x - width / 2 * scale_factor) ** 2 + (
#                                        y - height / 2 * scale_factor) ** 2) <= radius_outer ** 2]
#         if len(coords) >= num_users:
#             distances = [None for i in range(num_users)]
#             for index, tup in enumerate(coords[:num_users]):
#                 distances[index] = np.sqrt(tup[0] ** 2 + tup[1] ** 2) / 1000
#             return coords[:num_users], distances
#         scale_factor *= 1.1  # 增加比例因子尝试生成更多点


def path_loss_bs2client(d):
    return 128.1 + 37.6 * np.log10(d)


def path_loss_p2p(d):
    return 148 + 40 * np.log10(d)


def SNR_calculation(distances, num_users, flag):
    # Calculate path loss for each channel
    # if flag == 1 为计算 bs 2 client  ==0 为 p2p
    if flag == 1:
        path_loss_dB = path_loss_bs2client(distances)
    else:
        path_loss_dB = path_loss_p2p(distances)
    # Apply log-normal shadowing (8 dB standard deviation)    正态分布，标准差为10   size为生成的个数
    shadowing = np.random.normal(scale=8, size=num_users)
    # 计算大尺度衰落
    large_scale_fading = path_loss_dB + shadowing
    # Convert large_scale_fading from dB to linear scale
    large_scale_fading_linear = np.power(10, -large_scale_fading / 10)

    # Generate Rayleigh fading coefficients (complex numbers)
    fading = np.random.rayleigh(size=num_users) * np.exp(2j * np.pi * np.random.rand(num_users))

    # 计算复数幅度的均值和标准差
    magnitude = np.abs(fading)
    mean_magnitude = np.mean(magnitude)
    std_magnitude = np.std(magnitude)

    # 将衰落信号矫正为零均值和单位方差
    magnitude_normalized = (magnitude - mean_magnitude) / std_magnitude
    fading_normalized = magnitude_normalized * np.exp(1j * np.angle(fading))

    # 计算SNR线性值
    channel_coefficients = fading_normalized * np.sqrt(large_scale_fading_linear)
    # 计算信道增益的模的平方
    gain = abs(channel_coefficients) ** 2
    # 计算snr 并转化成dB形式 （p*||G||**2）/ (B* No)
    # 放大系数 p/B*N
    scaling_factor = 1.5 * 10 ** 14.4
    # SNR
    snr_b2c = gain * scaling_factor
    snr_b2c = 10 * np.log10(snr_b2c)
    # 输出是dB 值，在进行速率计算时，要改成线性值
    return snr_b2c

# 把样本的sample id传进来
# num_samples 是每个用户所需的数据量
def cifar_iid_exchange(dataset, sample_index, num_samples, client_id):

    data_num = len(dataset)
    class_num = 10
    idxs = sample_index
    # 将排序后的 NumPy 数组转换为 Python 列表
    idxs_list = idxs.tolist()
    labels = np.array(dataset.targets)
    labels = labels[idxs_list]
    dict_users = {}

    overalldist = [0] * class_num
    for i in range(len(labels)):
        overalldist[labels[i]] += 1
    overalldist.insert(0, 0)
    for i in range(1, class_num + 1):
        overalldist[i] += overalldist[i - 1]
    heads = overalldist.copy()
    del [heads[class_num]]

    # sort labels  ⭐
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    dominances = []
    counts = {}
    for i in range(class_num):
        counts[i] = 0

    for i in client_id:
        subset, domi = dominance_client(heads, overalldist, idxs, counts, sampleNum=num_samples, dominance=0,
                                        classNum=class_num)
        dict_users[i] = subset
        dominances.append(domi)

    return dict_users


def normalize_np(np_list):
    min_np = np.min(np_list)
    max_np = np.max(np_list)
    if max_np - min_np == 0:
        raise ValueError("数组中的所有元素都相同，无法进行归一化。")
    normalized_arr = (np_list - min_np) / (max_np - min_np)
    return normalized_arr


if __name__ == '__main__':
    '''
    dataset_train = datasets.MNIST('../../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 200
    d,domi = nclass_skewness_mnist(dataset_train, num)
    spell_partition(d,dataset_train.train_labels.numpy(),domi)
    '''
    
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    num = 200
    #d,domi = uni_skewness_cifar(dataset_train, num,num_samples=2000)
    #d = dirichlet_skewness_cifar(dataset_train, num,num_samples=2000)
    d,domi = dirichlet_skewness_cifar(dataset_train, num,num_samples=2000)
    spell_partition(d,np.array(dataset_train.targets),domi)
    spell_data_usage(d,50000)


