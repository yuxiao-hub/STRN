"""CMU GCN architecture matching the June 2024 STRN checkpoint.

This module keeps the checkpoint's two graph-convolution branches and uses
the 75-coordinate CMU skeleton adjacency.  The extra modules at the end of
``GCN.__init__`` are retained because they are present in the checkpoint,
although the historical forward path does not call them.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from model import MultiAttention
from utils import util


CMU_DIM_USED = np.array([
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    75, 76, 77, 78, 79, 80, 84, 85, 86, 90, 91, 92,
    93, 94, 95, 96, 97, 98, 102, 103, 104, 105, 106,
    107, 111, 112, 113,
])


def get_dimused_adjacency(adjacency):
    return adjacency[:, CMU_DIM_USED][CMU_DIM_USED, :]


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
        (0, 3), (0, 8), (0, 13), (0, 9), (0, 14),
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16), (15, 17), (15, 21),
        (15, 30), (17, 18), (18, 19), (21, 22), (22, 23),
        (23, 25), (23, 28), (23, 24), (24, 25), (25, 26),
        (30, 31), (31, 32), (32, 34), (34, 35), (32, 37),
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
    def __init__(self, in_features, out_features, bias=True, node_n=75):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))

        adjacency = flatten_xyz_adjacency(normalize_adjacency(get_adjacency(38)))
        self.att2 = torch.from_numpy(get_dimused_adjacency(adjacency)).float()

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
        skeletal_adjacency = self.att2.to(device=inputs.device, dtype=inputs.dtype)
        skeletal_graph = torch.matmul(skeletal_adjacency, skeletal_support)
        output = self.a * learned_graph + self.b * skeletal_graph
        return output + self.bias if self.bias is not None else output


class GCBlock(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=75):
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
    def __init__(self, input_feature, hidden_feature, p_dropout,
                 num_stage=12, node_n=75, kernel_size=10):
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

        # Serialized by the 2024 training code, but unused by forward().
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
        idct_m = torch.from_numpy(idct_m).to(device=inputs.device, dtype=inputs.dtype)
        return torch.matmul(
            idct_m[:, :dct_n].unsqueeze(dim=0),
            output[:, :, :dct_n].transpose(1, 2),
        )
