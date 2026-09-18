import torch.nn as nn
import torch
import math

import torch


# class TransformerLayer(nn.Module):
#     def __init__(self, d_model, nhead):
#         super(TransformerLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.feedforward = nn.Sequential(
#             nn.Linear(d_model, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, d_model)
#         )
#         self.norm2 = nn.LayerNorm(d_model)
#
#     def forward(self, x):
#         # Self-Attention
#         attn_output, _ = self.self_attn(x, x, x)
#         # Residual Connection and Layer Normalization
#         x = x + attn_output
#         x = self.norm1(x)
#
#         # Feedforward
#         ff_output = self.feedforward(x)
#         # Residual Connection and Layer Normalization
#         x = x + ff_output
#         x = self.norm2(x)
#
#         return x


# # Example usage:
# d_model = 256
# nhead = 8
# transformer_layer = TransformerLayer(d_model, nhead)
# input_data = torch.randn(32, 20, d_model)  # Batch size: 16, Sequence length: 10, Feature dimension: d_model
# output_data = transformer_layer(input_data)
# print(output_data.shape)
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        # self.feedforward = nn.Sequential(
        #     nn.Linear(d_model, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, d_model)
        # )
        # self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.self_attn(x, x, x)
        # Residual Connection and Layer Normalization
        x = x + attn_output
        x = self.norm1(x)
        #
        # # Feedforward
        # ff_output = self.feedforward(x)
        # # Residual Connection and Layer Normalization
        # x = x + ff_output
        # x = self.norm2(x)

        return x