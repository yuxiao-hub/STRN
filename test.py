from utils import h36motion3d as datasets

from utils.opt import Options

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch

import numpy as np
import torch.nn.functional as F
import seaborn as sns


def main(opt):
    # opt.is_eval = True
    print('>>> create models')

    # acts = ["walking", "eating", "smoking", "discussion", "directions",
    #         "greeting", "phoning", "posing", "purchases", "sitting",
    #         "sittingdown", "takingphoto", "waiting", "walkingdog",
    #         "walkingtogether"]
    acts = ["walking"]
    for i, act in enumerate(acts):
        # i从0到14,遍历所有动作类别数据集
        test_dataset = datasets.Datasets(opt, split=2, actions=[act])
        test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True)

        run_model(data_loader=test_loader)


def calculate_similarity_matrix(input):
    a = input / torch.norm(input, dim=-1, keepdim=True)
    similarity = torch.matmul(a, a.permute(0, 2, 1))
    return similarity


def calculate_attentionmatrix(input):
    similarity = torch.matmul(input, input.permute(0, 2, 1))
    return similarity


def run_model(data_loader=None):
    n = 0
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

    for i, (p3d_h36) in enumerate(data_loader):
        # p3d_h36 [32, 75, 96]

        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        # if batch_size == 1 and is_train == 0:
        #     continue

        n += batch_size

        p3d_h36 = p3d_h36.float().cuda()
        p3d_src = p3d_h36.clone()[:, :, dim_used]

        # p3d_src [32, 75, 66]
        input2 = p3d_src.permute(0, 2, 1)
        print("input2_shape={}".format(input2.shape))
        simality_matrix = calculate_attentionmatrix(input2)
        min_val = simality_matrix.min()  # 原始数据的最小值
        max_val = simality_matrix.max()  # 原始数据的最大值
        # simality_matrix = F.softmax(simality_matrix, dim=-1)
        normalized_data = (simality_matrix - min_val) / (max_val - min_val)
        # ---------------visual_map_________________#

        # attention_matrix = simality_matrix[0].cpu().numpy()  # 提取第一个batch并转到CPU再转换为numpy
        attention_matrix = normalized_data[0].cpu().numpy()  # 提取第一个batch并转到CPU再转换为numpy
        # 提取每个关节的 x 维（即每隔3个元素取一次）
        x_indices = np.arange(0, 66, step=3)  # 选择第1, 4, 7, ..., 63
        x_values = attention_matrix[x_indices]  # 提取对应位置的值
        y_values = x_values[:, x_indices]
        min_val = y_values.min()  # 原始数据的最小值
        max_val = y_values.max()  # 原始数据的最大值
        normalized_data = (y_values - min_val) / (max_val - min_val)
        print(y_values.shape)

        # # 构建新的 [22, 22] 矩阵
        # new_matrix = x_values.reshape(22, 22)

        # 使用 seaborn 或 matplotlib 绘制热图
        plt.figure(figsize=(8, 8))
        sns.heatmap(y_values, cmap='viridis', square=True, cbar=True, xticklabels=10, yticklabels=10)
        plt.title('Attention Matrix (Batch 1)')
        plt.xlabel('Joint Index')
        plt.ylabel('Joint Index')
        plt.show()


if __name__ == '__main__':
    option = Options().parse()
    main(option)
