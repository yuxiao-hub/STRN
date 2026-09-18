from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import utils.util as util
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# SFOS模块
def calculate_predict_frames(input_x):
    # 计算沿时间维度（dim=2）的帧差值
    frame_diff = torch.diff(input_x, dim=2)
    # [frame_diff] [32, 66, 19]
    window_size = 4
    num_predictions = 10
    window = frame_diff[:, :, -window_size:]
    predictions_frames = []
    start_frame = input_x[:, :, -1:]
    for _ in range(window_size):
        pred = window.mean(dim=-1, keepdim=True)  # [32, 66, 1]
        next_frame = start_frame + pred
        predictions_frames.append(next_frame)
        start_frame = next_frame
        # 更新窗口，将最新的预测帧加入窗口，移除最旧的一帧
        window = torch.cat((window[:, :, 1:], pred), dim=-1)  # [32, 66, 6]

    predictions = torch.cat(predictions_frames, dim=-1)  # [32, 66, 6]

    predictions_frames = predictions[:, :, window_size - 1:window_size]
    repeated_frames = predictions_frames.repeat(1, 1, num_predictions - window_size)
    predictions = torch.cat([predictions, repeated_frames], dim=-1)
    return predictions


class AttModel(Module):
    # 子序列计算注意力与加权融合

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=12, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        self.kernel_size = kernel_size
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10
        # 查询向量q
        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   # in_features * T -> d_model * (T - kernel_size + 1)
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   # d_model * (T - kernel_size + 1) -> d_model * ((T - kernel_size + 1) -
                                   # kernel_size + 1)
                                   nn.ReLU())
        # 查询向量k
        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features, kernel_size=kernel_size)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        Training:
        src :Tensor(32, 60, 66) [batch_size,seq_len,feat_dim]
        Tensor(32, 75, 66)
        output_n: 10
        input_in: 50
        itera: 3

        """

        dct_n = self.dct_n

        src = src[:, :input_n]  # [bs,in_n,dim]

        src_tmp = src.clone()
        bs = src.shape[0]  # batch_size = 32
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()
        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n

        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])

        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,31,1320]

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n

        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)
        # 迭代预测，一次预测10帧
        for i in range(itera):

            query_tmp = self.convQ(src_query_tmp / 1000.0)

            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15

            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))

            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])

            input_gcn = src_tmp[:, idx]

            last_half = input_gcn.permute(0, 2, 1)  # [32, 66, 20]
            kernel_frames = last_half[:, :, :10]
            predicted_frames = calculate_predict_frames(kernel_frames)
            input_gcn = torch.cat([kernel_frames, predicted_frames], dim=-1)
            input_gcn = input_gcn.permute(0, 2, 1)  # [32, 20, 66]

            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)

            # gcn模块计算
            out_gcn = self.gcn(dct_in_tmp, is_out_resi=True, output_n=output_n, dct_n=dct_n)
            # 逆DCT变换＋截取后kernel_size+output_n大小的序列用作最后的预测，从[32, 66, 40]->[32, 20, 66]

            outputs.append(out_gcn.unsqueeze(2))
            if itera > 1:
                # update key-value query
                out_tmp = out_gcn.clone()[:, 0 - output_n:]
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)

                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        outputs = torch.cat(outputs, dim=2)
        return outputs
