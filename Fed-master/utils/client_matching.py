import random

import numpy
import numpy as np
import math
import networkx as nx


# 考虑：建立一个字典，给每个ch生成一个client对
# 可以先生成图 （顶点和边）  再添加权重
def generate_graph(num_users, ch_set, num_rrb_resources, coordinates, radius_clients):
    G = nx.Graph()
    existing_nodes = set()
    # 添加节点,这里的（i,k,m）代表一个元组，效果等同于索引下标
    for k in ch_set:
        # 这里需要筛选掉ch本身
        for i in range(num_users):
            if i not in ch_set:
                # 计算当前ch与各个i的距离
                distance = np.sqrt((coordinates[i][0]-coordinates[k][0]) ** 2 + (coordinates[i][1]-coordinates[k][1]) ** 2)
                if distance <= radius_clients:
                    for m in range(num_rrb_resources):
                        G.add_node((i, k, m), weight=0.0, rate=0.0)
                        existing_nodes.add((i, k, m))
                else:
                    continue

    for node in existing_nodes:
        i, k, m = node  # 解包当前节点

        # 给相同基站相同rrb的添加边
        for j in range(num_users):
            if j != i and (j, k, m) in existing_nodes:
                G.add_edge((i, k, m), (j, k, m))
        for M2 in ch_set:
            for N2 in range(num_rrb_resources):
                if not((k == M2) and (m == N2)):
                    if (i, M2, N2) in existing_nodes:
                        G.add_edge((i, k, m), (i, M2, N2))

    return G, existing_nodes


# 给图中所有节点计算权重,包含 emd, 能量损耗 , ||p0-pu||
# 传入一个顶点元组，计算其weight
# delta 为阈值，在此阈值之后改变weight计算规则, delta这里需要调整一下
# 每个ch期望收到5个子用户，期望
def weights_calculate(emd, fn, client_data_num, index, delta, coordinates, emd_all):
    # index为顶点的索引元组
    i, k, r = index
    Tl = 5
    Cn = 800
    alpha = 10 ** (-28)
    p = 1  # 功率 1w
    dn = 8  # 模型大小 Kbit
    # ⭐这几个值没有通过自动导入进来(data_num)
    Emax = Tl * Cn * 1000 * alpha * (10**18) + (8000 / (1.07*10**7))
    Emin = Tl * Cn * 100 * alpha * ((0.0003*(10**9)) ** 2) + (8000 / (4.28*10**7))
    emd_max = max(emd_all)
    emd_min = min(emd_all)

    distance = np.sqrt((coordinates[i][0] - coordinates[k][0]) ** 2 + (coordinates[i][1] - coordinates[k][1]) ** 2) / 1000
    large_scale_fading = 148 + 40 * np.log10(distance) + 8
    large_scale_fading_linear = np.power(10, -large_scale_fading / 10)
    # Generate Rayleigh fading coefficients (complex numbers)
    fading = np.random.rayleigh(size=1) * np.exp(2j * np.pi * np.random.rand(1))

    # 计算SINR线性值
    channel_coefficients = fading * np.sqrt(large_scale_fading_linear)
    # 计算信道增益的模的平方
    gain = abs(channel_coefficients) ** 2
    # 计算sinr 并转化成dB形式 （p*||G||**2）/ (B* No)
    # 放大系数 p/B*N
    scaling_factor = 0.5 * 10 ** 14.4
    # SINR
    sinr_b2c = gain * scaling_factor
    v_comm = 2*(10**6) * np.log2(1 + sinr_b2c)
    E_i = Tl * Cn * client_data_num * alpha * (fn) ** 2 + (8000 / v_comm)
    E_tarnsto1 = 1.2*(E_i - Emin) / (Emax-Emin)
    emd_transto1 = (emd-emd_min) / (emd_max-emd_min)
    # 这里可以调超参数
    value_i = E_tarnsto1+emd_transto1
    return  value_i, v_comm

# 新，计算当前数据和整个组内数据适配度地
# 仅计算一个点的权重指数
# index 是 i,k,z-->client,ch,rrb
def weights_calculate_dataset(fn, client_data_num, index, coordinates, overall_client_data_population, num_classes, sub_dict, w1, w2):
    # index为顶点的索引元组
    i, k, r = index
    Tl = 5
    Cn = 800
    alpha = 10 ** (-28)
    p = 1  # 功率 1w
    dn = 8  # 模型大小 Kbit
    # ⭐这几个值没有通过自动导入出来(data_num)
    Emax = Tl * Cn * 1000 * alpha * (10**18) + (8000 / (2.01*10**6))
    # 为了规避e——value太小做处理
    Emin = Tl * Cn * 100 * alpha * ((0.00000001*(10**9)) ** 2) + (8000 / (1.07*10**9))

    distance = np.sqrt((coordinates[i][0] - coordinates[k][0]) ** 2 + (coordinates[i][1] - coordinates[k][1]) ** 2) / 1000
    # 8 是shadowing
    large_scale_fading = 148 + 40 * np.log10(distance) + 8
    large_scale_fading_linear = np.power(10, -large_scale_fading / 10)
    # Generate Rayleigh fading coefficients (complex numbers)
    fading = np.random.rayleigh(size=1) * np.exp(2j * np.pi * np.random.rand(1))

    # 计算SINR线性值
    channel_coefficients = fading * np.sqrt(large_scale_fading_linear)
    # 计算信道增益的模的平方
    gain = abs(channel_coefficients) ** 2
    # 计算sinr （p*||G||**2）/ (B* No)

    if len(sub_dict[i]) == 1:
        # 若该用户不存在inter-cell 的干扰则它的放大系数为 p/B*N, 这里的值是手算出来的值
        scaling_factor = 0.5 * 10 ** 14.4
    else:
        sum_interference = 0
        for ch_neighbor in sub_dict[i]:
            if ch_neighbor == k:
                continue
            else:
                # 重新计算该用户和相邻ch之间的干扰信道增益用户计算干扰
                distance_d2d = np.sqrt(
                    (coordinates[i][0] - coordinates[ch_neighbor][0]) ** 2 + (coordinates[i][1] - coordinates[ch_neighbor][1]) ** 2) / 1000
                # 8 是shadowing
                large_scale_fading_a = 148 + 40 * np.log10(distance_d2d) + 8
                large_scale_fading_a_linear = np.power(10, -large_scale_fading_a / 10)
                # Generate Rayleigh fading coefficients (complex numbers)
                fading_n = np.random.rayleigh(size=1) * np.exp(2j * np.pi * np.random.rand(1))
                # 计算SINR线性值
                channel_coefficients = fading_n * np.sqrt(large_scale_fading_a_linear)
                # 计算信道增益的模的平方
                gain_n = abs(channel_coefficients) ** 2
                # p * ||G||**2  , 这里可以引入衰减系数 1 or 0.x
                sum_interference = sum_interference + 1.0 * 1000 * gain_n
        scaling_factor = 1000/(sum_interference+2*10**(-11.4))

    # SINR
    sinr_b2c = gain * scaling_factor
    sinr_db = 10 * math.log(sinr_b2c, 10)
    v_comm = 2*(10**6) * np.log2(1 + sinr_b2c)
    E_i = Tl * Cn * client_data_num * alpha * (fn) ** 2 + (8000 / v_comm)
    # 归一化[]
    if E_i >= Emax:
        E_tarnsto1 = 0.99
    elif E_i <= Emin:
        E_tarnsto1 = 0.01
    else:
        E_tarnsto1 = 2.5*(E_i - Emin) / (Emax-Emin)
        if E_tarnsto1 >= 1:
            E_tarnsto1 = 1

    # 计算加入当前ch后集群中数据均衡程度，用emd检测
    client_i = numpy.array(overall_client_data_population[i])
    ch = numpy.array(overall_client_data_population[k])
    new_total = ch + client_i
    sum_new = np.sum(new_total)
    normalized_new = new_total / sum_new
    normalized_new_list = list(normalized_new)
    EMD = sum(abs(num - 1 / num_classes) for num in normalized_new_list)
    # 这里可以调超参数
    if sinr_db < 0:
        value_i = 10000
    else:
        value_i = w1 * E_tarnsto1 + w2 * EMD
    return value_i, v_comm, sinr_db, [client_data_num, E_i, E_tarnsto1, sinr_db, EMD]



# sinrb2c是np数组存放各个client的信噪比  输入dB
# client 是一个数组存放当前需要计算的client
# rate 单位是bit/s
def transmission_rate(client, sinr_b2c):
    b = 2*(10**6)     # Mhz带宽
    rate_dict = {ch: float for ch in client}
    for ch_x in client:
        sinr_line = np.power(10, sinr_b2c[ch_x] / 10)
        rate = b * np.log2(1 + sinr_line)
        rate_dict[ch_x] = rate
    return rate_dict


def Tq_n_calculation(rate_dict, ch_set, dn, Tq):
    # 里面存放各ch对应的计算受限时延
    Tq_ch = {ch: float for ch in ch_set}
    # key 是ch, value 是速率
    for key, value in rate_dict.items():
        Tc2b = dn / value
        Tq_i = Tq - Tc2b
        Tq_ch[key] = Tq_i
    return Tq_ch