import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=128, dropout=0.1, future_len=20):
        super(TransformerTimeSeriesModel, self).__init__()
        self.d_model = d_model
        self.future_len = future_len

        # 输入层：线性层将输入映射到 Transformer 需要的维度
        self.input_fc = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer 编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_encoder_layers
        )

        # Transformer 解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_decoder_layers
        )

        # 输出层：将解码后的特征映射到输出的关节数
        self.output_fc = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        """
        src: 过去的帧序列，形状为 [batch_size, 40, 66]，即 [batch_size, 帧数, 关节数]
        tgt: 用于解码的未来帧序列，形状为 [batch_size, future_len, 66]
        """
        batch_size = src.size(0)

        # 将输入映射到 Transformer 所需的 d_model 维度
        src = self.input_fc(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt = self.input_fc(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))

        # 添加位置编码
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # 编码器处理过去帧
        memory = self.encoder(src)

        # 解码器预测未来帧
        output = self.decoder(tgt, memory)

        # 将 Transformer 解码器的输出映射到 66 个关节
        output = self.output_fc(output)

        return output


# 输入张量：历史帧序列 [32, 40, 66]
X = torch.rand(32, 40, 66)

# 目标张量：未来的帧序列 [32, 20, 66]，真实的未来帧（用于训练）
Y = torch.rand(32, 20, 66)

# 定义模型
model = TransformerTimeSeriesModel(input_dim=66, output_dim=66, d_model=512, nhead=8, num_encoder_layers=6,
                                   num_decoder_layers=6, dim_feedforward=128, dropout=0.1, future_len=20)

# 前向传播：未来帧预测
output = model(X, Y)

print(output.shape)  # 输出应为 [32, 20, 66]，即预测的未来 20 帧
