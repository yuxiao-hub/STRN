import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import torch
#
# # 定义动作类别
# acts = ["walking", "eating", "smoking", "discussion", "directions",
#         "greeting", "phoning", "posing", "purchases", "sitting",
#         "sittingdown", "takingphoto", "waiting", "walkingdog",
#         "walkingtogether"]
#
# # 假设有一些样本的类别标签
# # 这里使用整数表示类别，例如：0对应"walking"，1对应"eating"，依此类推
# # 示例标签，假设有 5 个样本
# labels = torch.tensor([0, 2, 1, 4, 3])  # 每个整数对应 acts 中的一个动作
#
# # 将标签转换为独热编码，类别数为 15
# num_classes = len(acts)
# one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
#
# # 打印独热编码结果
# print(one_hot_labels)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# 假设 simality_matrix 是已经计算出来的
batch_size = 32
joint_count = 66
simality_matrix = torch.randn(batch_size, joint_count, joint_count).cuda()  # 模拟生成一个类似的张量

# 计算softmax（假设你已经完成了这一部分）
simality_matrix = F.softmax(simality_matrix, dim=-1)

# 从 batch_size 中选取一个批次，比如第一个批次
attention_matrix = simality_matrix[0].cpu().numpy()  # 提取第一个batch并转到CPU再转换为numpy

# 使用 seaborn 或 matplotlib 绘制热图
plt.figure(figsize=(8, 8))
sns.heatmap(attention_matrix, cmap='viridis', square=True, cbar=True, xticklabels=10, yticklabels=10)
plt.title('Attention Matrix (Batch 1)')
plt.xlabel('Joint Index')
plt.ylabel('Joint Index')
plt.show()
