#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
from itertools import combinations
from matplotlib.patches import Circle
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib import rcParams
import matplotlib.pyplot as plt
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
# from shapely.ops import unary_union
import copy
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
# import openpyxl
import torchvision

import math
import json
import matplotlib.pyplot as plt
from utils.sampling import normalize_np, mnist_iid, cifar_iid, cifar_iid_exchange, complex_skewness_mnist, uni_skewness_cifar, pareto_skewness_cifar, dirichlet_skewness_cifar, inversepareto_skewness_cifar, staged_skewness_cifar, fewclass_uni_skewness_cifar, myfourclass_lab, SNR_calculation, path_loss_p2p, path_loss_bs2client, coordinate_generation, generate_grid_points_in_annulus
from utils.options import args_parser
from models.Update import LocalUpdate_prox, simulate_element_loss_and_recovery, LocalUpdate, SingleBgdUpdate, Serial_LocalUpdate, MiniFL_SingleBgdUpdate, LocalUpdate_moon
from models.Nets import cnncifar_new2_moon, MLP, CNNMnist, CNNCifar, CNNCifar_New, cnncifar_new2, mnist_model2
from models.Fed import FedAvg, FedAvgWithCmfl, Fed_personal, FedAvg_index_client
from models.test import test_img
from utils.client_matching import weights_calculate_dataset, generate_graph, weights_calculate, transmission_rate, Tq_n_calculation
from utils.evaluate import l2NormEvaluate, FA_round, Similarity_analysis
from models.Bound import estimateBounds
from models.Bandit import SelfSparringBandit, Rexp3Bandit, OortBandit
from models.dla_simple import SimpleDLA
import random

# 记录emd 修改更均匀的fl
# 弄清为甚么sinr有负值

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args_dict = vars(args)
    with open('./save/federated_parameter','w') as f:
        json.dump(args_dict, f)
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.device = torch.device('cuda:0')
    print(torch.cuda.is_available())
    # seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    # restore args
    if args.dataset == 'mnist':
        args.num_channels = 1
        allsamples = 60000
    if args.dataset == 'cifar':
        args.num_channels = 3
        allsamples = 50000

    # 这是用来均匀分配数据量的--可以用来指出这里的不足，没有数据量的标签漂移(若数据量大于均匀分配的数据量，则会将数据集洗牌后重新把数据划分出去  or   我们可以采取Gan生成的方式来重新划分)
    numsamples = int(allsamples/args.num_users)

    # 默认是-1 即自动为各个client分配数据量，就是如上所示均匀分，采用dirichlet也是没有数据量漂移的
    if args.num_data > 0:
        numsamples = args.num_data

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=False, transform=trans_mnist)
        # 这里是为了进行dirichlet数据可视化
        labels = np.array(dataset_train.targets)
        classes = dataset_train.classes
        # sample users
        if args.sampling == 'iid':
            # 传递训练集样本，总client数，各个client的数据量
            dict_users, dominance = mnist_iid(dataset_train, args.num_users, numsamples)
        elif args.sampling == 'uniform':
            dict_users, dominance = complex_skewness_mnist(dataset_train, args.num_users, numsamples)
        elif args.sampling == 'my_fourclass_client':
            dict_users, dominance, client_data_num, emd_all, data_value, overall_client_data_population = myfourclass_lab(args.dataset, dataset_train, args.num_users, args.num_classes, args.dir)
            with open('./save/client_data_num', 'w') as f:
                json.dump(client_data_num, f)
            with open('./save/emd_all', 'w') as f:
                json.dump(emd_all, f)
        else:
            exit("Bad argument: sampling")
    # 这里做了数据增强,在各个本地batch加载图片的时候才进行随机的剪裁or反转；各个client的数据样本数不变，但由于有变化性实际训练的量如同变多了
    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=False, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=False, transform=transform_test)
        # 这里是为了进行dirichlet数据可视化
        labels = np.array(dataset_train.targets)
        classes = dataset_train.classes
        if args.sampling == 'iid':
            dict_users, dominance = cifar_iid(dataset_train, args.num_users, numsamples)
        elif args.sampling == 'nclass':
            print("Dont use this IID")
            exit(0)
            #dict_users, dominance = nclass_skewness_cifar(dataset_train, args.num_users, numsamples)
        elif args.sampling == 'uniform':
            dict_users, dominance = uni_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'ipareto':
            dict_users, dominance = inversepareto_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'dirichlet':
            dict_users, dominance = dirichlet_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'pareto':
            dict_users, dominance = pareto_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'staged':
            dict_users, dominance = staged_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'fewclass':
            dict_users, dominance = fewclass_uni_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'my_fourclass_client':
            # overall_client_data_population 是一个字典，保存每个用户中的各个类别样本的数量
            dict_users, dominance, client_data_num, emd_all, data_value, overall_client_data_population = myfourclass_lab(args.dataset, dataset_train, args.num_users, args.num_classes, args.dir)
            with open('./save/client_data_num', 'w') as f:
                json.dump(client_data_num, f)
            with open('./save/emd_all', 'w') as f:
                json.dump(emd_all, f)
        else:
            exit("Bad argument: iid")
    else:
        exit('Error: unrecognized dataset')
    # dataset_train中存放的是一系列元组（图的tensor，其标签） 这里是取了第1张图的图的tensor来看其形状, 用于之后自动生成模型
    img_size = dataset_train[0][0].shape

    # ⭐用iid去测试时，要把在sample中将emd设置成全0； 做随机选择ch和用户匹配实验时，要更改后续的挑选方案
    # 为各个client随机生成坐标, (radius, num_of_users)
    coordinates, distances = coordinate_generation(800, args.num_users)
    # 计算各个client的b2c sinr
    distances_np = np.array(distances)
    # snr_b2c 是np数组
    snr_b2c = SNR_calculation(distances_np, args.num_users, args.flag_sinr)
    # 计算各个client的 ch价值，注意此时 emd和data_value为 list, sinr为np数组

    # ⭐求各个client范围内的从属设备
    sub_client = {i: [] for i in range(args.num_users)}
    dis_sub = np.array([None for i in range(args.num_users)])
    # 存放各节点可接触到的数据总量
    data_sub = [None for i in range(args.num_users)]
    snr_sub = [None for i in range(args.num_users)]

    emd_all_new = [None for i in range(args.num_users)]
    for i in sub_client:
        for j in sub_client:
            distance_sub = np.sqrt(
                 (coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2)
            if distance_sub <= args.radius_clients and i != j:
                sub_client[i].append(j)
    # α 1/emd + β data_value + γ sinr/10
    for i in range(args.num_users):
        dis_sub[i] = np.array(overall_client_data_population[i])
        num_sum_sub = client_data_num[i]
        snr_sub[i] = snr_b2c[i]
        for j in sub_client[i]:
            dis_sub[i] = dis_sub[i] + np.array(overall_client_data_population[j])
            num_sum_sub += client_data_num[j]
            snr_sub[i] += snr_b2c[j]
        emd_all_new[i] = sum(abs((num / num_sum_sub) - 1 / args.num_classes) for num in dis_sub[i])
        data_sub[i] = num_sum_sub
        snr_sub[i] = snr_sub[i] / len(sub_client[i])

    # 改 原来是 emd_all
    emd_np = np.array(emd_all_new)
    emd_value_np = 1 / emd_np
    emd_value_np = normalize_np(emd_value_np)
    # 改 原来是 data_value 改成两者结合
    data_value_np = np.array(data_sub) + np.array(data_value)
    data_value_np = 0.5 * normalize_np(data_value_np)
    for i in range(args.num_users):
        if client_data_num[i] == 1000:
            data_value_np[i] += 0.25
        else:
            data_value_np[i] += 0.10
    # 改 原来是 snr_b2c
    snr_b2c_value_np = normalize_np(snr_sub)
    for i in range(args.num_users):
        if snr_b2c[i] <= 12.0:
            snr_b2c_value_np[i] = 0
    # 改：把value改成每个用户通信范围内所有用户的均值，data_value和snr可直接求平均，emd需要把所有的用户的sample重新计数

    # value_all 为各个用户进行ch选择的价值 (⭐alpha belta gamma 所在地！！！！)
    # 尝试进行归一化，每个值都进行归一化处理
    value_all = args.A * data_value_np + args.B * emd_value_np + args.C * snr_b2c_value_np
    # 这里需要将各个client的序号顺序也更新
    ascending_indices = np.argsort(value_all)
    # 这里是按价值降序排列的client的id索引号
    descending_indices = ascending_indices[::-1]
    sorted_values_descending = value_all[descending_indices]

    # 创建ch集合
    # 给ch计算丢包率
    # 用来保存各个参与训练用户的 丢包率
    packet_loss = [None for i in range(args.num_users)]
    ch_set = set()
    if args.client_sel == 'Hierarchical_normal':
        random_nums_set = set(random.sample(range(args.num_users), args.ch_num))
        ch_set = random_nums_set
        for ch in ch_set:
            temp_loss = 0.065 * 20 * math.exp(-0.12 * snr_b2c[ch])
            if temp_loss > 1:
                temp_loss = 1
            packet_loss[ch] = temp_loss
    elif args.client_sel == 'Hierarchical':
        # counts_ch 指标用来统计已经加入ch集合数
        counts_ch = 1
        for i in range(args.num_users):
            if i == 0:
                ch_set.add(descending_indices[0])
            elif i >= 1 and counts_ch < args.ch_num:
            # 判断该用户和之前的用户是否有覆盖面积相交
                coor_i = coordinates[descending_indices[i]]
                all_meet_requirement = all(math.sqrt((coordinates[index_ch][0] - coor_i[0]) ** 2 + (coordinates[index_ch][1] - coor_i[1]) ** 2) >= (2 * args.radius_clients * args.theta) for index_ch in ch_set)
                if all_meet_requirement == True:
                    ch_set.add(descending_indices[i])
                    counts_ch = counts_ch + 1
            elif counts_ch == args.ch_num:
                break
        for ch in ch_set:
            temp_loss = 0.065 * 20 * math.exp(-0.12 * snr_b2c[ch])
            if temp_loss > 1:
                temp_loss = 1
            packet_loss[ch] = temp_loss

    print('ch is :', ch_set)

    # for ch in ch_set:
    #     print('ch snr is',snr_b2c[ch])
    # print(descending_indices)
    # print(sorted_values_descending)

    #计算各个ch之间的重复覆盖面积
    ch_coords = [coordinates[ch] for ch in ch_set]
    radius_ch = 200
    total_overlap_area = 0
    for (x0, y0), (x1, y1) in combinations(ch_coords, 2):
        d = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        if d >= 2 * radius_ch:
            overlap_area = 0
        elif d == 0:
            overlap_area = np.pi * radius_ch ** 2
        else:
            r2 = radius_ch ** 2
            alpha = 2 * np.arccos(d / (2 * radius_ch))
            overlap_area = r2 * (alpha - np.sin(alpha)) / 2
        total_overlap_area += overlap_area




    # 计算所有ch的平均data_num
    datanum_all = [data_sub[ch] for ch in ch_set]
    data_mean = sum(datanum_all) / (args.ch_num)


    # 计算平均EMD
    emd_ch_all = [emd_all_new[ch] for ch in ch_set]
    emd_mean = sum(emd_ch_all) / (args.ch_num)
    print('ch emd mean is', emd_mean)
    # 输出总的重叠面积
    print("总重叠面积:", total_overlap_area)
    print('ch data-all-mean is', data_mean)
    # 计算各个ch-bs的 最高传输速度、传输时间、传输能耗
    # 字典 记录ch:rate ch2bs
    ch_transmission_rate = transmission_rate(ch_set, snr_b2c)
    # 字典 记录 ch:Tq,i   之后匹配到此ch的client也能用
    ch_Tq_n = Tq_n_calculation(ch_transmission_rate, ch_set, args.dn, args.Tq)
    # 计算ch的 最优fn 之后更新进 fn_NEW中就不会变动
    fn_ch_star = {ch: float for ch in ch_set}
    for key, value in ch_Tq_n.items():
        fn_i = (args.Tl * args.Cn * 1500) / value
        fn_ch_star[key] = fn_i

    # # Set global font to Arial
    # rcParams['font.family'] = 'Arial'

# #----------------------------------------------------------------
#     # 需要各个client每个类别的数据量
#     # Suppose we have data like this:
#
#     # 随机抽取10个客户的数据
#     keys = random.sample(list(overall_client_data_population.keys()), 15)
#     sampled_data = {key: overall_client_data_population[key] for key in keys}
#
#     df = pd.DataFrame(sampled_data,
#                       index=['Label1', 'Label2', 'Label3', 'Label4', 'Label5', 'Label6', 'Label7', 'Label8', 'Label9',
#                              'Label10'])
#
#     fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=500)
#     colors = [(0, 0, 1, 0.8),  # 更改最后一它代表颜色的透明度
#               (0, 0.5, 0, 0.8),
#               (1, 0, 0, 0.8),
#               (0, 1, 1, 0.8),
#               (1, 0, 1, 0.8),
#               (0.5, 0.5, 0, 0.8),
#               (0, 0.5, 0.5, 0.8),
#               (0.5, 0, 0.5, 0.8),
#               (0.5, 0.5, 0.5, 0.8),
#               (0, 0, 0, 0.8)]
#
#     df.transpose().plot(kind='bar', stacked=True, ax=ax, color=colors)
#
#     # Configure grid
#     ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
#     ax.set_axisbelow(True)
#
#     # 显式设置图例不可见
#     ax.legend().set_visible(False)  # 显式隐藏图例
#     ax.set_title('(a) Dirichlet Distribution and Quantity Skew   β=0.1', fontsize=8)
#     ax.set_xlabel("Client ID", fontsize=8)
#     ax.set_ylabel("Number of Data Samples", fontsize=8)
#
#     plt.xticks(rotation='horizontal', fontsize=8)
#     plt.yticks(fontsize=8)
#     plt.tight_layout()
#     plt.savefig("stacked_bar_chart2.png", dpi=600, bbox_inches='tight')
#
#----------------------------------------------------------------------#

    # # 设置全局字体为Arial
    # # ch绘图
    # matplotlib.rcParams['font.family'] = 'Arial'
    # matplotlib.rcParams['font.sans-serif'] = 'Arial'
    #
    # radius_outer = 800  # 设置大的外圈半径
    # radius_ch = 200  # CH 点的圆圈的半径，根据你的数据调整
    #
    # fig, ax = plt.subplots(figsize=(1.6, 1.6), dpi=600)
    #
    # # 绘制所有clients的位置
    # x_coords, y_coords = zip(*coordinates)
    # ax.scatter(x_coords, y_coords, color='blue', s=0.5, label='Clients')
    #
    # # 添加第一个 CH 点来确保图例可以创建，我们将其设置为透明，因此在图中看不见
    # ax.scatter([], [], color='red', s=0.5, label='CH')
    #
    # circle_outer = Circle((0, 0), radius_outer, color='black', fill=False, linewidth=0.8)
    # ax.add_patch(circle_outer)
    #
    # # 绘制ch_set中的client，使用不同颜色
    # for ch in ch_set:
    #     ax.scatter(coordinates[ch][0], coordinates[ch][1], color='red', s=0.5)
    #     # 为每个 CH 点画一个深灰色虚线圆圈
    #     circle_ch = Circle(coordinates[ch], radius_ch, color='darkgray', linestyle='dashed', fill=False, linewidth=0.5)
    #     circle_ch.set_clip_path(circle_outer)
    #     ax.add_patch(circle_ch)
    #
    #
    # # 设置图像边界
    # ax.set_xlim(-1000, 1000)
    # ax.set_ylim(-1000, 1000)
    #
    # # # 打开/关闭自动均匀刻度标记
    # # ax.set_aspect('equal')
    #
    # # 设置各个方面的字体大小
    # # ax.legend(fontsize=6, handlelength=2, handletextpad=2, borderaxespad=2, loc='upper right', bbox_to_anchor=(1.05, 1.05))
    # if args.ch_selection_way == 0:
    #     ax.set_title('(a) The Selection of CH Through Random Choice', fontsize=4)
    # else:
    #     ax.set_title('(b) The Selection of CH Through HEO-CCS', fontsize=4)
    # # ax.set_xlabel('X Coordinate', fontsize=8)
    # # ax.set_ylabel('Y Coordinate', fontsize=8)
    # # ax.tick_params(axis='both', which='both', direction='in', labelsize=5)
    #
    # # 隐藏边框
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # # 隐藏刻度
    # ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    #
    # # 添加虚线grid
    # # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # # ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    # # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    #
    # plt.savefig('sample_figure.png', dpi=600, bbox_inches='tight')  # 保存图像为600 dpi的分辨率
    #
    # # 显示图形
    # plt.show()

# ---------------------------------------------------------------------------------------

    # 这是我的算法，hierarchical_normal是普通层级算法
    if args.client_sel == 'Hierarchical':
        # 执行用户匹配

        # 创建图, 初始权重均为0, existing_nodes中存放所有的顶点，通过索引该集合来计算权重
        graph, existing_nodes = generate_graph(args.num_users, ch_set, args.num_rrb_resources, coordinates, args.radius_clients)

        # 初始化fn = fmax = 1Ghz
        fn_old = np.array([0.0 for i in range(args.num_users)])
        fn_new = np.array([10**9 for i in range(args.num_users)])
        for ch in ch_set:
            fn_new[ch] = fn_ch_star[ch]

        # ⭐求各个client是否属于inter_cell区间，此外求inter_cell区间内的节点的从属字典，分别属于哪几个ch
        subordination_dict = {i: [] for i in range(args.num_users) if i not in ch_set}
        for i in subordination_dict:
            for ch in ch_set:
                distance_d2d = np.sqrt(
                    (coordinates[i][0] - coordinates[ch][0]) ** 2 + (coordinates[i][1] - coordinates[ch][1]) ** 2)
                if distance_d2d <= args.radius_clients:
                    subordination_dict[i].append(ch)
        # 统计受干扰的设备数量
        count_dis = 0
        for key, value in subordination_dict.items():
            # 检查列表的长度是否大于或等于2
            if len(value) >= 2:
                # 如果是，增加计数器
                count_dis += 1
        # 统计可参与fl的所有数据量
        all_participate_data = 0
        for key, value in subordination_dict.items():
            # 检查列表的长度是否大于或等于2
            if len(value) >= 1:
                all_participate_data = all_participate_data + client_data_num[key]
        print('能参与fl的数据总量', all_participate_data)
        # print(subordination_dict)
        # 输出满足条件的列表数量
        print(f"有 {count_dis} 个受干扰的设备")

        # 计算权重
        t = 1
        while not(np.allclose(fn_old, fn_new, atol=1e-1)) and t < args.T_max_:
            optimal_G = nx.Graph()  # 创建最优图
            delta_count = 0
            # 每个点分别计算权重
            for vertex in existing_nodes:
                i, k, r = vertex
                # 这里的rate_i是 client 2 ch 的传输速率
                # 这里个计算value里面需要手动调参以及调整超参数
                value, rate_i, sinr_db, TEST1 = weights_calculate_dataset(fn_new[i], client_data_num[i], vertex, coordinates, overall_client_data_population, args.num_classes, subordination_dict, args.w1, args.w2)
                # print(f'当前是第{t}轮下{i}节点的数据：',TEST1)
                temp_loss_2 = 0.065 * 20 * math.exp(-0.12 * sinr_db)
                if temp_loss_2 > 1:
                    temp_loss_2 = 1
                packet_loss[i] = temp_loss_2
                # print('丢包率是:', packet_loss[i])
                graph.nodes[vertex]['weight'] = value
                graph.nodes[vertex]['rate'] = rate_i
            # 按升序排
            sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[1]['weight'])
            # print(sorted_nodes)
            # 选择节点, 没有写在最后几轮更换value的方法，先保持如此⭐
            # optimal_G中保存有已经确定加入的用户集合，可以通过这个计算当前集群的总体数据样本均衡度
            graph_stay = graph.copy()
            while sorted_nodes:
                current_node, current_attrs = sorted_nodes.pop(0)  # 选择排序后第一个节点
                i1, k1, r1 = current_node
                # 更新overall_client_data_population
                overall_client_data_population[k1] = [a + b for a, b in zip(overall_client_data_population[k1], overall_client_data_population[i1])]
                neighbors = list(graph_stay.neighbors(current_node))  # 找到与当前节点相邻的节点
                # 检查是否符合Tq，就是验证满足Tq的fn有没有超过1Ghz

                # 将当前节点加入新图，并复制其属性
                optimal_G.add_node(current_node, **current_attrs)
                delta_count = delta_count+1
                # 删除当前节点及其相邻节点
                graph_stay.remove_node(current_node)
                for neighbor in neighbors:
                    if neighbor in graph_stay:
                        graph_stay.remove_node(neighbor)

                # 更新排序后的节点列表（去除已删除的节点）
                sorted_nodes = [node for node in sorted_nodes if node[0] in graph_stay]
                # 更新权重，包括重新计算以及排序⭐(所有与k有关的都需要计算,包括overall_client_data_population[ch])
                if args.flag_pruning == 0:
                    nodes_with_k1 = [node for node in graph_stay.nodes() if node[1] == 'k1']
                    for x in range(len(nodes_with_k1)):
                        i2, k2, z2 = x
                        value, rate_i = weights_calculate_dataset(fn_new[i2], client_data_num[i2], x, coordinates, overall_client_data_population, args.num_classes)
                        graph_stay.nodes[x]['weight'] = value
                        graph_stay.nodes[x]['rate'] = rate_i
                    sorted_nodes = sorted(graph_stay.nodes(data=True), key=lambda x: x[1]['weight'])


            # 更新fn，这里仅更新client ，仅计算有变化的   把new存到老的， 改fn_new
            fn_old = copy.deepcopy(fn_new)
            for node in optimal_G:
                if isinstance(node, tuple) and len(node) == 3:
                    i, k, r = node
                    weight = optimal_G.nodes[node]['weight']
                    rate = optimal_G.nodes[node]['rate']
                    x1 = (args.Tl * args.Cn * client_data_num[k]) / ch_Tq_n[k]
                    x2 = (args.Tl * args.Cn * client_data_num[i]) / (ch_Tq_n[k] - (2*args.dn / rate))
                    if x1 < x2:
                        fn_client_star = x1
                    else:
                        fn_client_star = x2
                    # test
                    if fn_client_star < 600000:
                        fn_client_star = 600000
                    fn_new[i] = fn_client_star
            t = t+1
        # 经过如上的步骤会得到一个optimal_G最优图，以及最优的 fn_new 根据此图进行后面的分层fl
        # 生成一个字典，保存 ch : [当前ch的子设备]
        management = { ch : [] for ch in ch_set}
        for node in optimal_G:
            if isinstance(node, tuple) and len(node) == 3:
                i, k, r = node
                management[k].append(i)
                # print(node)
        # print(management)
        fn_new = fn_new.tolist()



        # 检测每个簇的数据均衡程度,把簇内的所有数据都累加起来(算各个簇内的平均值)
        # 算整个参与fl 的 我的层级算法的平均数据均衡都需要统计数据量的和
        overall_data_mean = []
        for key, value in management.items():
            sum_array = np.zeros(args.num_classes, dtype=int)
            value.append(key)
            for po in value:
                sum_array = sum_array + np.array(overall_client_data_population[po])
            sum_1 = np.sum(sum_array)
            data_mean = sum_array / sum_1 if sum_1 else np.zeros(args.num_classes)
            data_mean = data_mean - 0.1
            data_mean = np.abs(data_mean)
            delta_data = np.sum(data_mean)

            overall_data_mean.append(float(delta_data))

            # print(overall_data_mean)
        total_data_mean = sum(overall_data_mean) / args.ch_num
        overall_data_mean.append(total_data_mean)
        with open('./save/overall_data_mean', 'w') as f:
            json.dump(overall_data_mean, f)

        # emddata-charge
        # ⭐数据交换
        if args.data_charge == 1:
            # 每个簇单独算
            print('----------------data_charge_on------------------')
            # dict_users_new 原本为空字典
            dict_users_new = {}
            list_ch =[]
            for key, value in management.items():
                sample_num_all = 0
                ch_sample_all = dict_users[key]
                num_ch_clients = len(value)
                sample_num_all = len(dict_users[key])
                dict_users_new[key] = dict_users[key]
                # 整理该簇内所有的用户样本资源
                for client in value:
                    sample_num_all = sample_num_all + len(dict_users[client])
                    ch_sample_all = np.concatenate([ch_sample_all, dict_users[client]])
                    # 这里用户没有分配额外的数据，其实可以配平⭐
                num_samples_clients = math.floor(sample_num_all / num_ch_clients)
                dict_users_now = cifar_iid_exchange(dataset_train, ch_sample_all, num_samples_clients, value)
                list_ch.append(key)
                more_sample = int(client_data_num[key]*1.2)
                new_ch = cifar_iid_exchange(dataset_train, ch_sample_all, more_sample, list_ch)
                dict_users_new.update(dict_users_now)
                dict_users_new.update(new_ch)
            dict_users = dict_users_new
            # for key,value in dict_users.items():
            #     print(len(value))
                # print('key is', key)


        # 计算单轮总能耗⭐  2 ch-client + ch_j +client_j + ch 2 bs 传
        e_ch2client_sum = 0.0
        e_ch2bs_sum = 0.0
        e_ch_j_sum = 0.0
        e_client_j_sum = 0.0
        alpha = 10 ** (-28)
        for node in optimal_G:
            if isinstance(node, tuple) and len(node) == 3:
                i, k, r = node
                rate = optimal_G.nodes[node]['rate']
                # 2 ch-client
                e_ch2client_sum = e_ch2client_sum + (2*8000) / rate

                e_client_j_sum = e_client_j_sum + args.Tl * args.Cn * client_data_num[i] * alpha * (fn_new[i])**2

        for ch in ch_set:
            e_ch2bs_sum = e_ch2bs_sum + (3*8000) / ch_transmission_rate[ch]
            e_ch_j_sum += args.Tl * args.Cn * client_data_num[ch] * alpha * (fn_new[ch])**2

        E_total_for_one_round = e_ch2client_sum + e_ch2bs_sum + e_ch_j_sum + e_client_j_sum
        E_total_for_one_round = E_total_for_one_round[0]
        print('E_one_epoch is', E_total_for_one_round)
        E_total = E_total_for_one_round * args.epochs
        energy = [E_total_for_one_round, E_total]
        print(energy)
        with open('./save/energy_consumption_jiaoer', 'w') as f:
            json.dump(energy, f)

    # build model
    if args.model == 'resnet50' and args.dataset == 'cifar':
        net_glob = torchvision.models.resnet50(pretrained=False)
    elif args.model == 'vgg16' and args.dataset == 'cifar':
        net_glob = torchvision.models.vgg16(pretrained=False)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = torchvision.models.resnet18(pretrained=False)
    elif args.model == 'sdla' and args.dataset == 'cifar':
        net_glob = SimpleDLA().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        print('ok')
    elif args.model == 'cnn_new' and args.dataset == 'cifar':
        net_glob = CNNCifar_New(args=args).to(args.device)
    elif args.model == 'cnncifar_new2' and args.dataset == 'cifar':
        net_glob = cnncifar_new2(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnncifar_new2_moon' and args.dataset == 'cifar':
        net_glob = cnncifar_new2_moon(args=args).to(args.device)
    elif args.model == 'mnist_model2' and args.dataset == 'mnist':
        net_glob = mnist_model2(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print('网络模型是：', net_glob)



    # 训练着手启动
    net_glob.train()

    # client selection scheme
    if args.client_sel == 'random':
        pass
    elif args.client_sel == 'fedacs':
        # ⭐在进行4类用户测试的时候在bandit模块有代码修改
        bandit = SelfSparringBandit(args)
    elif args.client_sel == 'rexp3':
        bandit = Rexp3Bandit(args)
    elif args.client_sel == 'oort':
        bandit = OortBandit(args)
    elif args.client_sel == 'Hierarchical':
        print('now is Hierarchical fl')
    elif args.client_sel == 'Hierarchical_normal':
        print('now is Hierarchical_normal fl')
    elif args.client_sel == 'cluster':
        print('now is cfl')
    elif args.client_sel == 'EARA':
        print('now is EARA')
    else:
        print("Bad Argument: client_sel")
        exit(-1)


    # copy weights
    w_glob = net_glob.state_dict()
    w_old = copy.deepcopy(w_glob)
    # 要至少保存两轮的模型
    previous_net = copy.deepcopy(net_glob.state_dict())

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    locallr = args.lr

    if args.testing > 0:
        testacc = []

    similog = []
    domilog = []
    rewardlog = []
    hitmaplog = []

    # 创建了一个 行为 epoch-1 列为num_users的 全为-1的二维阵
    l2eval = np.ones((args.epochs-1,args.num_users))
    l2eval = l2eval * -1

    removed = []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    # # ⭐这里是采用2.5层聚合重新给各个client分组的user_idx
    # if args.sampling == "my_fourclass_client":
    #     nums = list(range(20, 110))
    #     # 随机打乱列表
    #     random.shuffle(nums)
    #     # 创建字典存储8组随机数字，key为40-47  （now 9组）
    #     group_dict = {key: nums[i * 10: (i + 1) * 10] for i, key in enumerate(range(20, 29))}
    #     # 打印每组数据
    #     for key, values in group_dict.items():
    #         print(f"Group {key}: {values}")

    # # ⭐测试p概率数组，让0-19被选的几率减小
    # num_units = 48
    # low_range = list(range(0, 20))
    # high_range = list(range(20, num_units))
    #
    # low_prob = 0.15  # 为0-19分配的总概率
    # high_prob = 0.85  # 为20-47分配的总概率
    #
    # prob = np.zeros(num_units)
    # prob[low_range] = low_prob / len(low_range)
    # prob[high_range] = high_prob / len(high_range)

    #   开始  全局的各个epoch迭代
    #   确保数据也加载至gpu  eg
    #   inputs = torch.randn(100, 10).to(device)  # 100个样本，每个样本10个特征
    #   targets = torch.randn(100, 1).to(device)
    energy_consumption = []
    energy_consumption_accumulate = []

    # ⭐开始迭代
    for iter in range(1, args.epochs+1):
        # 开始 train
        loss_locals = []

        if args.frac > 1:
            m = int(args.frac)
        else:
            m = max(int(args.frac * args.num_users), 1)
        eva_locals = []
        # 用来检测本轮被抽样的用户群体的整体多样性
        eva_gradients = []
        loss_reward = {}
        # client selection
        if args.client_sel == 'oort' or args.client_sel == 'fedacs' or args.client_sel == 'rexp3':
            # 后期根据最优的beta分布，几乎只利用不探索，故后期框定的client就基本上是固定的那些client，有必要考虑怎么利用失败节点吗
            # ⭐注意采用我的四类用户的sample时修改代码
            idxs_users = bandit.requireArms(m)
        elif args.client_sel == 'Hierarchical':
            participate_c = []
            for node in optimal_G:
                if isinstance(node, tuple) and len(node) == 3:
                    participate_c.append(node[0])
            ch_set_np = np.array(list(ch_set))
            participate_c_np = np.array(participate_c)
            idxs_users = np.concatenate((ch_set_np, participate_c_np))
            print('所有参与计算的用户为:', idxs_users)
        elif args.client_sel == 'EARA':
            positive_indices = [i for i, elem in enumerate(snr_b2c) if elem >= 0]
            if len(positive_indices) < m:
                raise ValueError("Not enough non-negative elements to select from.")
            print(positive_indices)
            ch_set = np.random.choice(positive_indices, size=args.ch_num, replace=False)
            print(ch_set)
            ch_transmission_rate = transmission_rate(ch_set, snr_b2c)
            # 执行用户匹配
            for ch in ch_set:
                temp_loss = 0.065 * 20 * math.exp(-0.12 * snr_b2c[ch])
                if temp_loss > 1:
                    temp_loss = 1
                packet_loss[ch] = temp_loss

            # 创建图, 初始权重均为0, existing_nodes中存放所有的顶点，通过索引该集合来计算权重
            graph, existing_nodes = generate_graph(args.num_users, ch_set, args.num_rrb_resources, coordinates,
                                                   args.radius_clients)


            # ⭐求各个client是否属于inter_cell区间，此外求inter_cell区间内的节点的从属字典，分别属于哪几个ch
            subordination_dict = {i: [] for i in range(args.num_users) if i not in ch_set}
            for i in subordination_dict:
                for ch in ch_set:
                    distance_d2d = np.sqrt(
                        (coordinates[i][0] - coordinates[ch][0]) ** 2 + (coordinates[i][1] - coordinates[ch][1]) ** 2)
                    if distance_d2d <= args.radius_clients:
                        subordination_dict[i].append(ch)
            # 统计受干扰的设备数量
            count_dis = 0
            for key, value in subordination_dict.items():
                # 检查列表的长度是否大于或等于2
                if len(value) >= 2:
                    # 如果是，增加计数器
                    count_dis += 1
            # 统计可参与fl的所有数据量
            all_participate_data = 0
            for key, value in subordination_dict.items():
                # 检查列表的长度是否大于或等于2
                if len(value) >= 1:
                    all_participate_data = all_participate_data + client_data_num[key]
            print('能参与fl的数据总量', all_participate_data)
            # print(subordination_dict)
            # 输出满足条件的列表数量
            print(f"有 {count_dis} 个受干扰的设备")

            # 计算权重
            optimal_G = nx.Graph()  # 创建最优图
            delta_count = 0
            # 每个点分别计算权重
            for vertex in existing_nodes:
                i, k, r = vertex
                # 这里的rate_i是 client 2 ch 的传输速率
                # 这里个计算value里面需要手动调参以及调整超参数
                value, rate_i, sinr_db, TEST1 = weights_calculate_dataset(10 ** 9, client_data_num[i], vertex,
                                                                              coordinates,
                                                                              overall_client_data_population,
                                                                              args.num_classes, subordination_dict,
                                                                              args.w1, args.w2)
                # print(f'当前是第{t}轮下{i}节点的数据：',TEST1)
                temp_loss_2 = 0.065 * 20 * math.exp(-0.12 * sinr_db)
                if temp_loss_2 > 1:
                    temp_loss_2 = 1
                packet_loss[i] = temp_loss_2
                # print('丢包率是:', packet_loss[i])
                graph.nodes[vertex]['weight'] = value
                graph.nodes[vertex]['rate'] = rate_i
            # 按升序排
            sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[1]['weight'])
            # print(sorted_nodes)
            # 选择节点, 没有写在最后几轮更换value的方法，先保持如此⭐
            # optimal_G中保存有已经确定加入的用户集合，可以通过这个计算当前集群的总体数据样本均衡度
            graph_stay = graph.copy()
            while sorted_nodes:
                current_node, current_attrs = sorted_nodes.pop(0)  # 选择排序后第一个节点
                i1, k1, r1 = current_node
                # 更新overall_client_data_population
                overall_client_data_population[k1] = [a + b for a, b in zip(overall_client_data_population[k1],
                                                                                overall_client_data_population[i1])]
                neighbors = list(graph_stay.neighbors(current_node))  # 找到与当前节点相邻的节点
                # 检查是否符合Tq，就是验证满足Tq的fn有没有超过1Ghz

                # 将当前节点加入新图，并复制其属性
                optimal_G.add_node(current_node, **current_attrs)
                delta_count = delta_count + 1
                # 删除当前节点及其相邻节点
                graph_stay.remove_node(current_node)
                for neighbor in neighbors:
                    if neighbor in graph_stay:
                        graph_stay.remove_node(neighbor)

                # 更新排序后的节点列表（去除已删除的节点）
                sorted_nodes = [node for node in sorted_nodes if node[0] in graph_stay]
                # 更新权重，包括重新计算以及排序⭐(所有与k有关的都需要计算,包括overall_client_data_population[ch])
                if args.flag_pruning == 0:
                    nodes_with_k1 = [node for node in graph_stay.nodes() if node[1] == 'k1']
                    for x in range(len(nodes_with_k1)):
                        i2, k2, z2 = x
                        value, rate_i = weights_calculate_dataset(10 ** 9, client_data_num[i2], x, coordinates,
                                                                      overall_client_data_population, args.num_classes)
                        graph_stay.nodes[x]['weight'] = value
                        graph_stay.nodes[x]['rate'] = rate_i
                    sorted_nodes = sorted(graph_stay.nodes(data=True), key=lambda x: x[1]['weight'])
            participate_c = []
            # 经过如上的步骤会得到一个optimal_G最优图，以及最优的 fn_new 根据此图进行后面的分层fl
            # 生成一个字典，保存 ch : [当前ch的子设备]
            management = {ch: [] for ch in ch_set}
            for node in optimal_G:
                if isinstance(node, tuple) and len(node) == 3:
                    i, k, r = node
                    management[k].append(i)
                    # print(node)
            # print(management)
            for key, value in management.items():
                participate_c.append(key)
                for i in value:
                    participate_c.append(i)
            idxs_users = np.array(participate_c)
            # 计算单轮总能耗⭐  2 ch-client + ch_j +client_j + ch 2 bs 传
            e_ch2client_sum = 0.0
            e_ch2bs_sum = 0.0
            e_ch_j_sum = 0.0
            e_client_j_sum = 0.0
            alpha = 10 ** (-28)
            for node in optimal_G:
                if isinstance(node, tuple) and len(node) == 3:
                    i, k, r = node
                    rate = optimal_G.nodes[node]['rate']
                    # 2 ch-client
                    e_ch2client_sum = e_ch2client_sum + (2 * 8000) / rate

                    e_client_j_sum = e_client_j_sum + args.Tl * args.Cn * client_data_num[i] * alpha * (10**9) ** 2

            for ch in ch_set:
                e_ch2bs_sum = e_ch2bs_sum + (3 * 8000) / ch_transmission_rate[ch]
                e_ch_j_sum += args.Tl * args.Cn * client_data_num[ch] * alpha * (10**9) ** 2

            E_total_for_one_round = e_ch2client_sum + e_ch2bs_sum + e_ch_j_sum + e_client_j_sum
            E_total_for_one_round = E_total_for_one_round[0]
            print('E_one_epoch is', E_total_for_one_round)
            energy_consumption.append(E_total_for_one_round)
            print('energy_consumption is', E_total_for_one_round)
        elif args.client_sel == 'Hierarchical_normal':
            # 记录各个ch可支配的用户
            if iter == 1:
                 ch_client_dict = {i:[] for i in range(args.num_users)}
                 for i in range(args.num_users):
                     for j in range(args.num_users):
                         if i != j:
                             distance = np.sqrt((coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2)
                             if distance <= args.radius_clients:
                                 ch_client_dict[i].append(j)
            # 每轮随机选ch  各ch随机选组内用户
            participate_c = []
            # 管理2大队
            management = {}
            # 去掉sinr低于0的用户⭐
            positive_indices = [i for i, elem in enumerate(snr_b2c) if elem >= 0]
            if len(positive_indices) < m:
                raise ValueError("Not enough non-negative elements to select from.")
            ch_random = np.random.choice(positive_indices, size=args.ch_num, replace=False)
            ch_transmission_rate = transmission_rate(ch_random, snr_b2c)
            for ch in ch_random:
                if len(ch_client_dict[ch]) < args.num_rrb_resources:
                    client_now = np.random.choice(ch_client_dict[ch], size=len(ch_client_dict[ch]), replace=False)
                else:
                    client_now = np.random.choice(ch_client_dict[ch], size=args.num_rrb_resources, replace = False)
                client_now_list = client_now.tolist()
                management[ch] = client_now_list
            for key, value in management.items():
                participate_c.append(key)
                for i in value:
                    participate_c.append(i)
            idxs_users = np.array(participate_c)

            # 检测每个簇的数据均衡程度,把簇内的所有数据都累加起来(算各个簇内的平均值)
            # 算整个参与fl 的 我的层级算法的平均数据均衡都需要统计数据量的和
            overall_data_mean = []
            for key, value in management.items():
                sum_array = np.zeros(args.num_classes, dtype=int)
                value.append(key)
                for po in value:
                    sum_array = sum_array + np.array(overall_client_data_population[po])
                sum_1 = np.sum(sum_array)
                data_mean = sum_array / sum_1 if sum_1 else np.zeros(args.num_classes)
                data_mean = data_mean - 0.1
                data_mean = np.abs(data_mean)
                delta_data = np.sum(data_mean)

                overall_data_mean.append(float(delta_data))
                print(overall_data_mean)
                print('ok')
            total_data_mean = sum(overall_data_mean) / args.ch_num
            overall_data_mean.append(total_data_mean)
            with open('./save/overall_data_mean', 'w') as f:
                json.dump(overall_data_mean, f)
        elif args.client_sel == 'random':
            # star
            # 去掉sinr低于0的用户⭐
            positive_indices = [i for i, elem in enumerate(snr_b2c) if elem >= 0]
            if len(positive_indices) < m:
                raise ValueError("Not enough non-negative elements to select from.")
            idxs_users = np.random.choice(positive_indices, size=m, replace=False)
            for i in idxs_users:
                temp_loss = 0.065 * 20 * math.exp(-0.12 * snr_b2c[i])
                if temp_loss > 1:
                    temp_loss = 1
                packet_loss[i] = temp_loss
        elif args.client_sel == 'cluster':
            # 创建一个新的列表来存放过滤后的结果
            filtered_a = []
            # 遍历a中的每个子列表
            for sublist in cluster_participate:
                # 创建一个新的子列表来存放满足条件的元素
                new_sublist = []
                # 遍历子列表中的每个元素
                for element in sublist:
                    # 检查b中对应索引的值是否大于等于0.5
                    if snr_b2c[element] >= 0.1:
                        # 如果是，则将该元素添加到新的子列表中
                        new_sublist.append(element)
                # 如果新的子列表不为空，则将其添加到过滤后的结果中
                if new_sublist:
                    filtered_a.append(new_sublist)
            print('过滤后的', filtered_a)
            idxs_users = [random.choice(sublist) for sublist in filtered_a]
            print('本轮迭代用户：', idxs_users)
            for i in idxs_users:
                temp_loss = 0.065 * 20 * math.exp(-0.12 * snr_b2c[i])
                if temp_loss > 1:
                    temp_loss = 1
                packet_loss[i] = temp_loss
        else:
            if args.sampling == 'fewclass':
                idxs_users = [random.randint(i, i+9) for i in range(0, 100, 10)]
                idxs_users = np.array(idxs_users)
            # # ⭐这里注释掉是为了测试使用random采样acc的曲线
            # elif args.sampling == "my_fourclass_client":
            #     # 先进行随机采样测试
            #     idxs_users = np.random.choice(range(0,29), m, replace=False)



        # if args.client_sel != 'Hierarchical':
        #     for i in idxs_users:
        #         print(i)
        #         print(snr_b2c[i])

        # 计算星型\普通层级fl能耗
        if args.client_sel != 'Hierarchical':
            # 星型fl
            if args.client_sel == 'random':
                e_ch2bs_sum = 0.0
                e_ch_j_sum = 0.0
                alpha = 10**(-28)
                fedavg_transmission_rate = transmission_rate(idxs_users, snr_b2c)
                for ch in idxs_users:
                    e_ch2bs_sum = e_ch2bs_sum + (3 * 8000) / fedavg_transmission_rate[ch]
                    e_ch_j_sum = e_ch_j_sum + args.Tl * args.Cn * client_data_num[ch] * alpha * (10**9) ** 2
                E_total_for_one_round = e_ch2bs_sum + e_ch_j_sum
                energy_consumption.append(E_total_for_one_round)
                print('energy_consumption is', E_total_for_one_round)
            elif args.client_sel == 'cluster':
                e_ch2bs_sum = 0.0
                e_ch_j_sum = 0.0
                alpha = 10**(-28)
                fedavg_transmission_rate = transmission_rate(idxs_users, snr_b2c)
                for ch in idxs_users:
                    e_ch2bs_sum = e_ch2bs_sum + (3 * 8000) / fedavg_transmission_rate[ch]
                    e_ch_j_sum = e_ch_j_sum + args.Tl * args.Cn * client_data_num[ch] * alpha * (10**9) ** 2
                E_total_for_one_round = e_ch2bs_sum + e_ch_j_sum
                energy_consumption.append(E_total_for_one_round)
                print('energy_consumption is', E_total_for_one_round)
            elif args.client_sel == 'EARA':
                pass
            # 层级fl
            else:
                # 计算单轮总能耗⭐  2 ch-client + ch_j +client_j + ch 2 bs 传
                # ch-client 通讯能耗
                e_ch2client_sum = 0.0
                # ch-bs 通讯
                e_ch2bs_sum = 0.0
                # ch 计算能耗
                e_ch_j_sum = 0.0
                # client 计算能耗
                e_client_j_sum = 0.0
                alpha = 10 ** (-28)
                # 计算ch-client之间的传输耗能及计算耗能
                for i in participate_c:
                    a = [i]
                    rate_dict = transmission_rate(a, snr_b2c)
                    rate = rate_dict[i]
                    # 2 ch-client
                    e_ch2client_sum = e_ch2client_sum + (2 * 8000) / rate
                    e_client_j_sum = e_client_j_sum + args.Tl * args.Cn * client_data_num[i] * alpha * (
                    0.95*(10**9)) ** 2

                for ch in ch_random:
                    e_ch2bs_sum = e_ch2bs_sum + (3 * 8000) / ch_transmission_rate[ch]
                    e_ch_j_sum = e_ch_j_sum + args.Tl * args.Cn * client_data_num[ch] * alpha * (0.95*(10**9)) ** 2

                E_total_for_one_round = e_ch2client_sum + e_ch2bs_sum + e_ch_j_sum + e_client_j_sum
                energy_consumption.append(E_total_for_one_round)
                print('E_one_epoch is', E_total_for_one_round)


        # 注意这里在每次迭代都会刷新，但是cluster不能刷新
        if args.aggregate_way == 2:
            w_locals = [None] * 48
        elif args.aggregate_way == 3:
            w_locals = [None] * args.num_users
        elif args.client_sel == 'cluster':
            if iter == 1:
                # 建立rrb个可保存的模型
                w_locals_cluster = [copy.deepcopy(net_glob.state_dict())] * args.num_rrb_resources
            w_locals = []
        else:
            w_locals = []

        # print(type(idxs_users))
        # ⭐进行测试
        # 层次聚合的时候， 由于能耗和时间都是计算出来的，就是多了一层聚合  （ch和client都要参与计算）
        for idx in idxs_users:
            # # ⭐locallr就是学习率，这里的LocalUpdate是定义了一个本地更新的类
            # if 0<= idx <= 19:
            #     local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],locallr=locallr)
            #     # newlr是因为这里采取了lr衰减的策略
            #     w, loss, newlr = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # else:
            #     # 这里需要把各个小联邦中的client的数据串起来组成 字典 传入 idxs中
            #     client_sample = { id : [] for id in group_dict[idx]}
            #     for id in group_dict[idx]:
            #         client_sample[id] = dict_users[idx]
            #     local = Serial_LocalUpdate(args=args, dataset=dataset_train, idxs=client_sample, locallr=locallr)
            #     w, loss, newlr = local.train(copy.deepcopy(net_glob).to(args.device), group_dict[idx])
            if args.fed_algorithm == 2 and iter >= 2:
                local = LocalUpdate_moon(args=args, dataset=dataset_train, idxs=dict_users[idx], locallr=locallr)
                w, loss, newlr = local.train(net=copy.deepcopy(net_glob).to(args.device), previous_nets = previous_net)
            elif args.fed_algorithm == 1:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], locallr=locallr)
                w, loss, newlr = local.train(net=copy.deepcopy(net_glob).to(args.device), flag=args.fed_algorithm)
            elif args.fed_algorithm == 3:
                local = LocalUpdate_prox(args=args, dataset=dataset_train, idxs=dict_users[idx], locallr=locallr)
                w, loss, newlr = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # 簇类要训练 r'r'b 个模型
            elif args.fed_algorithm == 4:
                if iter == 1:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], locallr=locallr)
                    w, loss, newlr = local.train(net=copy.deepcopy(net_glob).to(args.device), flag=args.fed_algorithm)
                else:
                    # 找到当前用户对应的簇类模型
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], locallr=locallr)
                    net_glob.load_state_dict(w_locals_cluster[clusters[idx]])
                    w, loss, newlr = local.train(net=copy.deepcopy(net_glob).to(args.device), flag=args.fed_algorithm)
            # newlr是因为这里采取了lr衰减的策略

            # print(w)
            # w_old 是上一轮的全局模型
            # ⭐测试考虑loss丢包的情形
            w = simulate_element_loss_and_recovery(w_old, w, packet_loss[idx], args.device)
            # ⭐按用户id存储模型
            if args.aggregate_way == 2:
                w_locals[idx] = copy.deepcopy(w)
            elif args.aggregate_way == 3:
                w_locals[idx] = copy.deepcopy(w)
            elif args.aggregate_way == 0:
                if args.fed_algorithm == 4:
                    w_locals_cluster[clusters[idx]] = copy.deepcopy(w)
                    w_locals.append(copy.deepcopy(w))
                else:
                    w_locals.append(copy.deepcopy(w))

            # loss的考虑？
            loss_locals.append(copy.deepcopy(loss))
            loss_reward[idx] = loss


        # update global weights
        if args.aggregate_way == 0 or args.aggregate_way == 4:
            w_glob = FedAvg(w_locals)
        elif args.aggregate_way == 2:
            # fed_personal是按照用户id 储存本地模型的
            w_glob = Fed_personal(w_locals, idxs_users)
        elif args.aggregate_way == 3:
        # 先对client 进行聚合
            w_array = []
            if args.client_sel == 'Hierarchical':
                for ch in ch_set:
                    w_array.append(FedAvg_index_client(w_locals, management, ch))
            else:
                for ch in ch_random:
                    w_array.append(FedAvg_index_client(w_locals, management, ch))
            # 再对ch进行聚合
            print(len(w_array))
            w_glob = FedAvg(w_array)
        else:
            w_glob = FedAvgWithCmfl(w_locals, copy.deepcopy(w_old))

        previous_net = copy.deepcopy(w_old)
        w_old = copy.deepcopy(w_glob)
        # copy weight to net_glob，加载至模型上
        net_glob.load_state_dict(w_glob)
        net_glob.cuda()

        print(next(net_glob.parameters()).device)

        # domirankrecord = []
        # for client in idxs_users:
        #     domirankrecord.append(domirank[client])
        # domirankrecord.sort()
        #
        # hitmaplog.append(domirankrecord)


        # Log domi   每轮的平均domi
        if iter % args.testing == 0 and args.testing > 0:
            # domi = []
            # for client in idxs_users:
            #     domi.append(dominance[client])
            # domilog.append(float(sum(domi)/len(domi)))

            # test and log accuracy
            
            net_glob.eval()
            # ⭐这里修改 dataset_test 写个循环maybe可以测得每个用户的本地acc
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Round {:3d}, Testing accuracy: {:.2f}".format(iter, acc_test))
            net_glob.train()
            # print("Dominance ranking: " + str(domirankrecord))
            if len(removed) > 0:
                print("Removed: " + str(removed))

            testacc.append(float(acc_test))

        # Learning rate decay，下一轮各个参与的client都将采取这个学习率进行学习
        locallr = newlr
        
    # =====Rounds terminals=====  全局迭代结束，存放相关结果
    logidx = str(args.log_idx)

    if args.log_idx < 0:
        filepath = "./save/acc.log"
    else:
        filepath = "./save/acc/"+logidx+'.log'

    if args.client_sel != 'Hierarchical':
        # cluster 还要加一个聚类开销
        if args.fed_algorithm == 4:
            e_ch2bs_sum = 0.0
            e_ch_j_sum = 0.0
            alpha = 10 ** (-28)
            all_client = [i for i in range(args.num_users)]
            fedavg_transmission_rate = transmission_rate(all_client, snr_b2c)
            for id in all_client:
                e_ch2bs_sum = e_ch2bs_sum + (3 * 8000) / fedavg_transmission_rate[id]
                e_ch_j_sum = e_ch_j_sum + args.Tl * args.Cn * client_data_num[id] * alpha * (10 ** 9) ** 2
            E_total_for_start = e_ch2bs_sum + e_ch_j_sum
            list_accumulate_now = E_total_for_start
            for i in energy_consumption:
                list_accumulate_now += i
                energy_consumption_accumulate.append(list_accumulate_now)
            energy_consumption.append(sum(energy_consumption)+E_total_for_start)
            with open('./save/federated_fedavg_energy', 'w') as f:
                json.dump(energy_consumption, f)
            with open('./save/federated_fedavg_energy_accumulate', 'w') as f:
                json.dump(energy_consumption_accumulate, f)
        else:
            list_accumulate_now = 0
            for i in energy_consumption:
                list_accumulate_now += i
                energy_consumption_accumulate.append(list_accumulate_now)
            energy_consumption.append(sum(energy_consumption))
            with open('./save/federated_fedavg_energy','w') as f:
                json.dump(energy_consumption, f)
            with open('./save/federated_fedavg_energy_accumulate','w') as f:
                json.dump(energy_consumption_accumulate, f)

    with open(filepath,'w') as fp:
        for i in range(len(testacc)):
            content = str(testacc[i]) + '\n'
            fp.write(content)
        print('Acc log has written')

    if args.log_idx < 0:
        filepath = "./save/simi.log"
    else:
        filepath = "./save/simi/" + logidx + '.log'

    with open(filepath, 'w') as fp:
        for i in range(len(similog)):
            content = str(similog[i]) + '\n'
            fp.write(content)
        print('similarity log has written')

    if args.log_idx < 0:
        filepath = "./save/reward.log"
    else:
        filepath = "./save/reward/"+logidx+'.log'

    with open(filepath,'w') as fp:
        for i in range(len(rewardlog)):
            content = str(rewardlog[i]) + '\n'
            fp.write(content)
        print('Reward log has written')


    if args.sampling == 'staged':
        dominance = np.expand_dims(dominance,0)
        l2eval = np.concatenate((dominance,l2eval),axis=0)
        # writer = pd.ExcelWriter('l2eval.xlsx')
        # l2pandas = pd.DataFrame(l2eval)
        # l2pandas.to_excel(writer, 'sheet_1', float_format='%f')
        # writer._save()
        # writer.close()
        # print('L2 record written')
    