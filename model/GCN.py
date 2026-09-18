#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import utils.util as util
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np
import torch.nn.functional as F
from model import MultiAttention
import matplotlib.pyplot as plt
import seaborn as sns


# 选用22个关节的三维xyz坐标，对于Human3.6M而言
def get_dimused_Adj(adjacency):
    # adjacency = np.zeros((33, 33))
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    B = adjacency[:, dim_used]
    final_adj = B[dim_used, :]
    return final_adj


# 归一化图卷积的拉普拉斯矩阵 D^(-1)AD^(-1)
def norm_Adjcency(adjacency):
    Dl = np.sum(adjacency, 0)
    print(Dl)
    num_node = adjacency.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, adjacency), Dn)
    return DAD


# 构造人体关节连接的图邻接矩阵，基于Human3.6M数据集
def get_Adjcency(num_node):
    self_link = [(i, i) for i in range(num_node)]
    neighbor_link_ = [(0, 2), (0, 7), (0, 12), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9),
                      (9, 10), (12, 13), (13, 14), (13, 17), (13, 25),
                      (14, 15), (16, 17), (17, 18), (18, 19),
                      (19, 21), (19, 22), (20, 22), (24, 25), (25, 26), (26, 27), (27, 29), (27, 30), (28, 29)]

    neighbor_link = [(i, j) for (i, j) in neighbor_link_]
    edge = self_link + neighbor_link
    adjacency = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            for (x, y) in edge:
                if (x, y) == (i, j):
                    adjacency[i][j] = 1
                    adjacency[j][i] = adjacency[i][j]
    return adjacency


# 沿xyz关节轨迹展开 [66, 66]
def get_xyz_flatten_adjacency(adjacency):
    num_node = adjacency.shape[0]
    A = np.zeros((num_node * 3, num_node * 3))
    for i in range(num_node):
        for j in range(num_node):
            A[i * 3][j * 3] = adjacency[i][j]
            A[i * 3][j * 3 + 1] = adjacency[i][j]
            A[i * 3][j * 3 + 2] = adjacency[i][j]
            A[i * 3 + 1][j * 3] = adjacency[i][j]
            A[i * 3 + 1][j * 3 + 1] = adjacency[i][j]
            A[i * 3 + 1][j * 3 + 2] = adjacency[i][j]
            A[i * 3 + 2][j * 3] = adjacency[i][j]
            A[i * 3 + 2][j * 3 + 1] = adjacency[i][j]
            A[i * 3 + 2][j * 3 + 2] = adjacency[i][j]
    return A


# 高斯核函数计算相似度，用作消融
def batch_gaussian_kernel_matrix(X, sigma=0.2):
    # X: [batch_size, num_joints, time_dim] -> [32, 66, 40]

    # 计算每个关节时间序列的 L2 范数（平方和），保持批处理
    X_norm = torch.sum(X ** 2, dim=2, keepdim=True)  # [32, 66, 1]

    # 计算两两之间的欧氏距离平方
    dist_squared = X_norm + X_norm.transpose(1, 2) - 2 * torch.bmm(X, X.transpose(1, 2))  # [32, 66, 66]

    # 计算高斯相似度
    similarity_matrix = torch.exp(-dist_squared / (2 * sigma ** 2))

    return similarity_matrix


# 余弦相似度计算相似度，时间和空间
def calculate_similarity_matrix(input):
    a = input / torch.norm(input, dim=-1, keepdim=True)
    similarity = torch.matmul(a, a.permute(0, 2, 1))
    return similarity


# 图卷积计算，三个不同图卷积特征融合
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    # 对于每一帧的dct特征表示为 66×(2*dct_n)

    # -------------------------------------------------------------------------------------------------------
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        # dctn*2 的dct系数长度
        self.out_features = out_features
        # out_features 是gcn的隐藏特征维度，为256
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # A * H * W的W 训练权重
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        # A * H * W的A 参数化的可学习邻接矩阵
        num_node = 33
        adjacency = get_Adjcency(num_node)
        adjacency_norm = norm_Adjcency(adjacency)
        adjacency_xyz_flatten = get_xyz_flatten_adjacency(adjacency_norm)
        self.att2 = torch.from_numpy(get_dimused_Adj(adjacency_xyz_flatten)).float().cuda()
        # A * H * W中的A 基于关节连接的图邻接矩阵A

        self.weight2 = Parameter(torch.FloatTensor(in_features, out_features))
        self.a = Parameter(torch.FloatTensor(1))
        # 用作二者特征融合的可学习系数a，对于参数化的可学习图卷积

        self.b = Parameter(torch.FloatTensor(1))
        # 用作二者特征融合的可学习系数b，对于基于关节连接的图卷积

        self.c = Parameter(torch.FloatTensor(1))
        # 用作二者特征融合的可学习系数b，对于基于相似度构造的图邻接矩阵的图卷积

        self.weight3 = Parameter(torch.FloatTensor(in_features, out_features))
        self.fc1 = nn.Linear(node_n, 128)
        self.fc2 = nn.Linear(128, node_n)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.reset_parameters2()

    # 对参数归一化初始化，方便梯度下降
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 对参数归一化初始化，方便梯度下降
    def reset_parameters2(self):
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
        # ---------------------------------------------------------------------------------------
        stdv1 = 1. / math.sqrt(self.weight3.size(1))
        self.weight3.data.uniform_(-stdv1, stdv1)
        # ----------------------------------------------------------------------------------------
        stdv2 = 1. / math.sqrt(self.b.size(0))
        self.b.data.uniform_(-stdv2, stdv2)
        stdv3 = 1. / math.sqrt(self.a.size(0))
        self.a.data.uniform_(-stdv3, stdv3)
        stdv4 = 1. / math.sqrt(self.c.size(0))
        self.c.data.uniform_(-stdv4, stdv4)

    def forward(self, input):
        # -------------------------------------------------------------------------

        # input:[32, 66, 40]

        # 计算余弦相似度
        simality_matrix = calculate_similarity_matrix(input)
        simality_matrix = F.softmax(simality_matrix, dim=-1)

        support3 = torch.matmul(input, self.weight3)
        output3 = torch.matmul(simality_matrix, support3)
        # 图卷积A*H*W操作，这是计算基于相似度的构造的图邻接矩阵的图卷积

        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        # 图卷积A*H*W操作，这是计算参数化可学习的图邻接矩阵的图卷积

        support2 = torch.matmul(input, self.weight2)
        output2 = torch.matmul(self.att2, support2)
        # 图卷积A*H*W操作，这是基于关节连接构造的图邻接矩阵的图卷积

        result = torch.add(torch.mul(self.a, output), torch.mul(self.b, output2))
        result = torch.add(torch.mul(self.c, output3), result)
        # 三者特征通过三个可学习参数融合

        if self.bias is not None:
            return result + self.bias
        else:
            return result

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# 残差连接块
class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        GC_Block里的GCN的特征维度不变，均为256
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


#
#
# class TemporalTransformer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_joints, num_heads=8, num_layers=2):
#         super(TemporalTransformer, self).__init__()
#         # 输入特征提升
#         self.input_projection = nn.Linear(input_dim, hidden_dim)
#
#         # 定义 Transformer 编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,  # 提升后的特征维度
#             nhead=num_heads,  # 注意力头数
#             dim_feedforward=hidden_dim * 4,  # FFN的隐藏层维度
#             dropout=0.4,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer, num_layers=num_layers
#         )
#
#         # 输出特征还原
#         self.output_projection = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         # 输入 x: [batch_size, num_joints, input_dim]
#         x = self.input_projection(x)  # [batch_size, num_joints, hidden_dim]
#         x = self.transformer_encoder(x)  # [batch_size, num_joints, hidden_dim]
#         x = self.output_projection(x)  # [batch_size, num_joints, output_dim]
#         return x
# #
#
# class TCN(nn.Module):
#     def __init__(self, input_length, feature_size, kernel_size):
#         super(TCN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=256, kernel_size=kernel_size, padding=2,
#                                dilation=2)
#
#         self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_size, padding=4, dilation=4)
#
#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, 66)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = torch.relu(x)
#         x = self.conv2(x)
#         x = torch.relu(x)
#
#         x = x.transpose(1, 2)
#
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         return x

#
# class T_GCN(nn.Module):
#     def __init__(self, time_dim, hidden_feature):
#         super(T_GCN, self).__init__()
#         self.fc1 = nn.Linear(time_dim, hidden_feature)
#         self.fc2 = nn.Linear(hidden_feature, time_dim)
#         self.matrix = torch.ones(time_dim, time_dim)
#
#         # 将其转化为下三角矩阵
#         self.lower_triangular_matrix = torch.tril(self.matrix).cuda()
#
#     def forward(self, x):
#         # x: [32, 20, 66]
#         channel_pool = torch.mean(x, dim=2, keepdim=True)
#         score = torch.matmul(channel_pool, channel_pool.permute(0, 2, 1))
#         y = self.fc1(score)
#         y = F.relu(y)
#         y = self.fc2(y)
#         y = F.relu(y)
#         score_norm = F.softmax(y, dim=1)
#         mask = self.lower_triangular_matrix * score_norm
#         output = torch.matmul(mask, x)
#         return output

# 时间图卷积分支
class Temporal_GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True, node_n=48, p_dropout=0.3):
        super(Temporal_GCN, self).__init__()
        # x:[32, 40, 66]
        self.in_features = in_features
        # 66
        self.out_features = out_features
        # 128
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # W:[66, 128]
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        # A;[40, 40]
        self.weight3 = Parameter(torch.FloatTensor(in_features, out_features))
        # [66, 128]
        self.fc = nn.Linear(out_features, in_features)

        self.bn1 = nn.BatchNorm1d(node_n * in_features)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.a = Parameter(torch.FloatTensor(1))

        self.b = Parameter(torch.FloatTensor(1))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters2()

    def forward(self, x):
        # 输入 x: [32, 40, 66]
        simality_matrix = calculate_similarity_matrix(x)
        simality_matrix = F.softmax(simality_matrix, dim=-1)
        # simality_matrix :[32, 40, 40]
        support3 = torch.matmul(x, self.weight3)
        output3 = torch.matmul(simality_matrix, support3)

        # -----------------------------------------------------------------------------------------------------

        support = torch.matmul(x, self.weight)

        # A*H*W
        output = torch.matmul(self.att, support)

        result = torch.add(torch.mul(self.a, output), torch.mul(self.b, output3))
        x = self.fc(result)
        return x

    def reset_parameters2(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.weight3.size(1))
        self.weight3.data.uniform_(-stdv1, stdv1)

        stdv2 = 1. / math.sqrt(self.b.size(0))
        self.b.data.uniform_(-stdv2, stdv2)
        stdv3 = 1. / math.sqrt(self.a.size(0))
        self.a.data.uniform_(-stdv3, stdv3)

# 空间图卷积分支
class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=12, node_n=48, kernel_size=10):
        """

        # input_feature = dct_n * 2 = 40
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature = 256
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks = 12
        :param node_n: number of nodes in graph = 66
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.node_n = node_n

        # input_feature 为dct_n*2= 40  hidden_feature为256  node_n为66  num_stage = 12
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.kernel_size = kernel_size

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        # 将num_stages个残差网络连接在一块
        self.gcbs = nn.ModuleList(self.gcbs)

        # 最后一层图卷积将隐藏层特征从256维转变为40维的姿态空间表示
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.temporal_gcn = Temporal_GCN(in_features=66, out_features=128, node_n=40, p_dropout=0.3)

        self.a = Parameter(torch.FloatTensor(1))

        self.b = Parameter(torch.FloatTensor(1))
        #
        # # ---------------------------------------------------------------------

    def reset_parameters2(self):
        stdv2 = 1. / math.sqrt(self.b.size(0))
        self.b.data.uniform_(-stdv2, stdv2)
        stdv3 = 1. / math.sqrt(self.a.size(0))
        self.a.data.uniform_(-stdv3, stdv3)

    def forward(self, x, is_out_resi=True, output_n=10, dct_n=20):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        if is_out_resi:
            y = y + x

        # y波浪线 [B，J，2(H+T)]
        # ----------------------------------------
        x = x.permute(0, 2, 1)
        # y2 = self.te_transformer(x)
        # y2 = x.permute(0, 2, 1)
        # y = torch.add(torch.mul(self.a, y2), torch.mul(self.b, y))
        x = self.temporal_gcn(x)
        y2 = x.permute(0, 2, 1)
        y = torch.add(torch.mul(self.a, y2), torch.mul(self.b, y))

        # ----------------------------------------------------------
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        y1 = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                          y[:, :, :dct_n].transpose(1, 2))
        # y1:[32, 20, 66]

        # -----------------------------------------------------------------------------

        # ---------------------------------------------------------------------------

        return y1
