"""CMU attention model wired to the recovered 2024 two-branch GCN."""

import numpy as np
import torch
from torch import nn

from model import GCN_CMU_2024_recovered
from utils import util


class AttModel(nn.Module):
    def __init__(self, in_features=75, kernel_size=10, d_model=256,
                 num_stage=12, dct_n=20):
        super().__init__()
        if kernel_size != 10:
            raise ValueError("The recovered checkpoint requires kernel_size=10.")
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n
        self.convQ = nn.Sequential(
            nn.Conv1d(in_features, d_model, kernel_size=6, bias=False),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, bias=False),
            nn.ReLU(),
        )
        self.convK = nn.Sequential(
            nn.Conv1d(in_features, d_model, kernel_size=6, bias=False),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, bias=False),
            nn.ReLU(),
        )
        self.gcn = GCN_CMU_2024_recovered.GCN(
            input_feature=dct_n * 2,
            hidden_feature=d_model,
            p_dropout=0.3,
            num_stage=num_stage,
            node_n=in_features,
            kernel_size=kernel_size,
        )

    def forward(self, src, output_n=25, input_n=50, itera=1):
        dct_n = self.dct_n
        src_tmp = src[:, :input_n].clone()
        batch_size = src_tmp.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, _ = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).to(device=src.device, dtype=src.dtype)

        value_count = input_n - self.kernel_size - output_n + 1
        value_length = self.kernel_size + output_n
        value_indices = (
            np.expand_dims(np.arange(value_length), axis=0)
            + np.expand_dims(np.arange(value_count), axis=1)
        )
        src_value_tmp = src_tmp[:, value_indices].clone().reshape(
            batch_size * value_count, value_length, -1
        )
        src_value_tmp = torch.matmul(
            dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp
        ).reshape(batch_size, value_count, dct_n, -1).transpose(2, 3).reshape(
            batch_size, value_count, -1
        )

        input_indices = list(range(-self.kernel_size, 0)) + [-1] * output_n
        outputs = []
        key_tmp = self.convK(src_key_tmp / 1000.0)

        for _ in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            attention = score_tmp / torch.sum(score_tmp, dim=2, keepdim=True)
            dct_attention = torch.matmul(attention, src_value_tmp)[:, 0].reshape(
                batch_size, -1, dct_n
            )

            # The 2024 checkpoint was trained with the standard HisRep input:
            # the last observed frame is repeated for the prediction interval.
            input_gcn = src_tmp[:, input_indices]
            dct_input = torch.matmul(
                dct_m[:dct_n].unsqueeze(dim=0), input_gcn
            ).transpose(1, 2)
            dct_input = torch.cat([dct_input, dct_attention], dim=-1)
            out_gcn = self.gcn(
                dct_input,
                is_out_resi=True,
                output_n=output_n,
                dct_n=dct_n,
            )
            outputs.append(out_gcn.unsqueeze(2))

            if itera > 1:
                src_tmp = torch.cat([src_tmp, out_gcn[:, -output_n:].clone()], dim=1)
                value_count = 1 - 2 * self.kernel_size - output_n
                value_length = self.kernel_size + output_n
                dct_indices = (
                    np.expand_dims(np.arange(value_length), axis=0)
                    + np.expand_dims(
                        np.arange(value_count, -self.kernel_size - output_n + 1),
                        axis=1,
                    )
                )
                src_key_tmp = src_tmp[:, dct_indices[0, :-1]].transpose(1, 2)
                key_tmp = torch.cat([key_tmp, self.convK(src_key_tmp / 1000.0)], dim=2)

                src_dct_tmp = src_tmp[:, dct_indices].clone().reshape(
                    batch_size * self.kernel_size, value_length, -1
                )
                src_dct_tmp = torch.matmul(
                    dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp
                ).reshape(
                    batch_size, self.kernel_size, dct_n, -1
                ).transpose(2, 3).reshape(batch_size, self.kernel_size, -1)
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)
                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        return torch.cat(outputs, dim=2)
