#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=500, help="number of users: K")
    parser.add_argument('--frac', type=float, default=8, help="the fraction of clients: C, when > 1, becomes number of clients selected")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnncifar_new2', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=2008, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    # modified
    parser.add_argument('--sampling', type=str, default='my_fourclass_client', help='iid || uniform || pareto || ipareto || dirichlet || staged || fewclass || my_fourclass_client')
    parser.add_argument('--testing', type=int, default= 1, help="test the model after some rounds, -1: never")

    parser.add_argument('--client_sel', type=str, default='EARA', help="Client selection, random || fedacs || rexp3 || oort || Hierarchical || Hierarchical_normal || clusters || EARA")
    parser.add_argument('--log_idx', type=int, default=-1, help="Index of log file")
    parser.add_argument('--faf', type=int, default=-1, help="How offen FA round is used, -1:never, 0:always, 1:early stop")
    parser.add_argument('--lrd', type=float, default=0.9993, help="Learning rate decay")
    parser.add_argument('--extension', type=int, default=8, help="Candidate extension")
    parser.add_argument('--num_data', type=int, default=-1, help="Number of data in each client, -1:auto")
    parser.add_argument('--historical_rounds', type=int, default=3, help="How many rounds are remenbered by bandit for historical comparison? 0: never")
    parser.add_argument('--cut_thres', type=float, default=0, help="Threshold to remove bad updates, 0: never")
    parser.add_argument('--aggregate_way', type=int, default=0, help='0 use fedavg , 1 use CMFL , 2 use fed_personal, 3 use Hierarchical average')
    parser.add_argument('--light_analytics', action='store_true', help='infer the skewness only based on the last layer')
    parser.add_argument('--radius', type=int, default=800, help='fl-radius')
    parser.add_argument('--ch_num', type=int, default=8, help='number of cluster header ')
    parser.add_argument('--radius_clients', type=int, default=200, help='client radius')
    parser.add_argument('--num_rrb_resources', type=int, default=8, help='num_rrb_resources')
    parser.add_argument('--delta_matching', type=int, default=30, help='num_rrb_resources')
    parser.add_argument('--Tq', type=int, default=1, help='time bounded')
    parser.add_argument('--dn', type=int, default=8000, help='model size 8000Kbit')
    parser.add_argument('--Tl', type=float, default=5, help='local learning epoch')
    parser.add_argument('--Cn', type=float, default=800, help='cpu_cycle')
    parser.add_argument('--T_max_', type=int, default=100, help='fn迭代次数max')
    parser.add_argument('--data_charge', type=int, default=0, help='0表示 不交换，1表示进行data均匀交换')

    parser.add_argument('--dir', type=float, default=0.1, help='表示dirichlet中的alpha')

    parser.add_argument('--ch_selection_way', type=int, default=0, help='1表示采用贪婪筛选，0表示随机')
    parser.add_argument('--flag_pruning', type=int, default=0, help='1表示采用复杂度减小技术，0表示不减小')
    parser.add_argument('--class_num_client', type=int, default=2, help='表示各个client中的样本种类数量')
    parser.add_argument('--A', type=float, default=0.5, help='表示CH选择中的超参数alpha--data')
    parser.add_argument('--B', type=float, default=0.5, help='表示CH选择中的超参数beta--emd')
    parser.add_argument('--C', type=float, default=0.35, help='表示CH选择中的超参数gamma--snr')
    parser.add_argument('--flag_sinr', type=int, default=1, help='1表示b2d 其他为d2d')
    parser.add_argument('--w1', type=float, default=0, help='表示 图匹配能耗权重')
    parser.add_argument('--w2', type=float, default=1.0, help='表示图匹配emd权重')
    parser.add_argument('--theta', type=float, default=0.8, help='表示可重叠比例')
    parser.add_argument('--fed_algorithm', type=int, default=1, help='1=normal  2=moon 3=fedprox 4=cluster ')
    args = parser.parse_args()
    return args
