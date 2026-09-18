"""Two-branch H3.6M GCN used by the June 25, 2024 STRN checkpoint."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from model import MultiAttention
from utils import util


def get_dimused_adjacency(adjacency):
    dim_used = np.array([
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66,
        67, 68, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92,
    ])
    return adjacency[:, dim_used][dim_used, :]


def normalize_adjacency(adjacency):
    degree = np.sum(adjacency, axis=0)
    inv_sqrt_degree = np.zeros((adjacency.shape[0], adjacency.shape[0]))
    for index, value in enumerate(degree):
        if value > 0:
            inv_sqrt_degree[index, index] = value ** -0.5
    return np.dot(np.dot(inv_sqrt_degree, adjacency), inv_sqrt_degree)


def get_adjacency(num_node):
    self_links = [(index, index) for index in range(num_node)]
    neighbor_links = [
        (0, 2), (0, 7), (0, 12), (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10), (12, 13), (13, 14), (13, 17),
        (13, 25), (14, 15), (16, 17), (17, 18), (18, 19), (19, 21),
        (19, 22), (20, 22), (24, 25), (25, 26), (26, 27), (27, 29),
        (27, 30), (28, 29),
    ]
    adjacency = np.zeros((num_node, num_node))
    for source, target in self_links + neighbor_links:
        adjacency[source, target] = 1
        adjacency[target, source] = 1
    return adjacency


def flatten_xyz_adjacency(adjacency):
    num_node = adjacency.shape[0]
    expanded = np.zeros((num_node * 3, num_node * 3))
    for row in range(num_node):
        for column in range(num_node):
            expanded[row * 3:row * 3 + 3, column * 3:column * 3 + 3] = adjacency[row, column]
    return expanded


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        adjacency = flatten_xyz_adjacency(normalize_adjacency(get_adjacency(33)))
        self.att2 = torch.from_numpy(get_dimused_adjacency(adjacency)).float().cuda()
        self.weight2 = Parameter(torch.FloatTensor(in_features, out_features))
        self.a = Parameter(torch.FloatTensor(1))
        self.b = Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-1.0, 1.0)
        self.b.data.uniform_(-1.0, 1.0)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        support = torch.matmul(inputs, self.weight)
        learned_graph = torch.matmul(self.att, support)
        skeletal_support = torch.matmul(inputs, self.weight2)
        skeletal_graph = torch.matmul(self.att2, skeletal_support)
        output = self.a * learned_graph + self.b * skeletal_graph
        return output + self.bias if self.bias is not None else output


class GCBlock(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)
        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        output = self.gc1(inputs)
        batch, nodes, features = output.shape
        output = self.bn1(output.view(batch, -1)).view(batch, nodes, features)
        output = self.dropout(self.activation(output))
        output = self.gc2(output)
        output = self.bn2(output.view(batch, -1)).view(batch, nodes, features)
        output = self.dropout(self.activation(output))
        return output + inputs


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=12, node_n=48, kernel_size=10):
        super().__init__()
        self.num_stage = num_stage
        self.kernel_size = kernel_size
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        self.gcbs = nn.ModuleList([
            GCBlock(hidden_feature, p_dropout=p_dropout, node_n=node_n)
            for _ in range(num_stage)
        ])
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.Tanh()

        # These modules were serialized by the June 2024 training code but
        # are not referenced by that version's GCN.forward implementation.
        # They are retained so the trusted checkpoint can load strictly.
        self.linear1 = nn.Linear(node_n, hidden_feature)
        self.linear2 = nn.Linear(hidden_feature, node_n)
        self.multi_head = MultiAttention.TransformerLayer(hidden_feature, 8)

    def forward(self, inputs, is_out_resi=True, output_n=10, dct_n=20):
        output = self.gc1(inputs)
        batch, nodes, features = output.shape
        output = self.bn1(output.view(batch, -1)).view(batch, nodes, features)
        output = self.dropout(self.activation(output))
        for block in self.gcbs:
            output = block(output)
        output = self.gc7(output)
        if is_out_resi:
            output = output + inputs

        _, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        idct_m = torch.from_numpy(idct_m).float().cuda()
        return torch.matmul(
            idct_m[:, :dct_n].unsqueeze(dim=0),
            output[:, :, :dct_n].transpose(1, 2),
        )
