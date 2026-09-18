import torch.nn as nn
import torch
import torch.nn.functional as F

#
# def calculate_predict_frames(input_x):
#     # 计算沿时间维度（dim=2）的帧差值
#     frame_diff = torch.diff(input_x, dim=2)
#     # [frame_diff] [32, 66, 19]
#     window_size = 4
#     num_predictions = 10
#     # 取最后4帧作为初始窗口
#     window = frame_diff[:, :, -window_size:]
#     # 存储预测结果
#     predictions_frames = []
#     start_frame = input_x[:, :, -1:]
#     for _ in range(num_predictions):
#         # 计算当前窗口的平均值
#         pred = window.mean(dim=-1, keepdim=True)  # [32, 66, 1]
#         next_frame = start_frame + pred
#         predictions_frames.append(next_frame)
#         start_frame = next_frame
#         # 更新窗口，将最新的预测帧加入窗口，移除最旧的一帧
#         window = torch.cat((window[:, :, 1:], pred), dim=-1)  # [32, 66, 4]
#     # 将所有预测帧拼接在一起
#     predictions = torch.cat(predictions_frames, dim=-1)  # [32, 66, 10]
#     return predictions
#
#
import torch

import torch
import torch.nn as nn


class SpatialTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_joints, num_heads=8, num_layers=2):
        super(SpatialTransformer, self).__init__()
        # 输入特征提升
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 定义 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # 提升后的特征维度
            nhead=num_heads,  # 注意力头数
            dim_feedforward=hidden_dim * 4,  # FFN的隐藏层维度
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 输出特征还原
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 输入 x: [batch_size, num_joints, input_dim]
        x = self.input_projection(x)  # [batch_size, num_joints, hidden_dim]
        x = self.transformer_encoder(x)  # [batch_size, num_joints, hidden_dim]
        x = self.output_projection(x)  # [batch_size, num_joints, output_dim]
        return x


# 参数设置
batch_size = 32
num_joints = 66
input_dim = 40  # 初始时间维度
hidden_dim = 128  # 增强后的特征维度
output_dim = 40  # 最终输出时间维度

# 创建模型
model = SpatialTransformer(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_joints=num_joints
)

# 输入张量
x = torch.rand(batch_size, num_joints, input_dim)  # [32, 66, 40]

# 前向传播
output = model(x)

print("输出形状:", output.shape)
# 高斯核函数计算批处理的相似度矩阵
# def batch_gaussian_kernel_matrix(X, sigma=1.0):
#     # X: [batch_size, num_joints, time_dim] -> [32, 66, 40]
#
#     # 计算每个关节时间序列的 L2 范数（平方和），保持批处理
#     X_norm = torch.sum(X ** 2, dim=2, keepdim=True)  # [32, 66, 1]
#
#     # 计算两两之间的欧氏距离平方
#     dist_squared = X_norm + X_norm.transpose(1, 2) - 2 * torch.bmm(X, X.transpose(1, 2))  # [32, 66, 66]
#
#     # 计算高斯相似度
#     similarity_matrix = torch.exp(-dist_squared / (2 * sigma ** 2))
#
#     return similarity_matrix
#
#
# # 输入张量 [32, 66, 40]，32 是 batch size，66 是关节数，40 是时间维度
# tensor = torch.randn(32, 66, 40)  # 使用随机数据作为示例
#
# # 计算相似度矩阵
# similarity_matrices = batch_gaussian_kernel_matrix(tensor, sigma=1.0)
#
# print(similarity_matrices.shape)  # 输出应该是 [32, 66, 66]
