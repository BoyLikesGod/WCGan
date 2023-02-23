import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.positional_encoding = Positional_Encoding(self.config.embedding_dim, self.config.pad_size,
                                                       self.config.gen_positional_dropout, self.config.device)
        self.encoder = Encoder(self.config.embedding_dim, self.config.gen_num_head,
                               self.config.gen_feed_forward_hidden_size,
                               self.config.gen_encoder_dropout, self.config.device)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(self.config.gen_num_encoder)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.positional_encoding(x)
        for encoder in self.encoders:
            out = encoder(out)

        out = self.sigmoid(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embedding_dim, pad_size, dropout, device):
        """
        :param embedding_dim:词向量维度
        :param pad_size:文本长度
        """
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (1000 ** (i // 2 * 2.0 / embedding_dim)) for i in range(embedding_dim)] for pos in
             range(pad_size)])  # (embedding_dim * pad_size * 1.0)
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=True).to(self.device)
        out = self.dropout(out)
        return out


class Position_wise_Feed_forward(nn.Module):
    def __init__(self, embedding_dim, hidden_size, dropout, device):
        """
        :param embedding_dim: 词向量维度
        :param hidden_size: 位置前馈网络中的隐藏层维度
        """
        super(Position_wise_Feed_forward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, embedding_dim).to(device)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim).to(device)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = x + out
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_head, hidden_size, dropout, device):
        """
        :param embedding_dim:词向量维度
        :param num_head:多头注意力中头的数量
        :param hidden_size:位置前馈网络中的隐藏层维度
        """
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(embedding_dim, num_head, dropout, device)
        self.feed_forward = Position_wise_Feed_forward(embedding_dim, hidden_size, dropout, device)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, device):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.device = device

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1)).to(self.device)
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1).to(self.device)
        context = torch.matmul(attention, V).to(self.device)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, embedding_dim, num_head, dropout, device):
        """
        :param embedding_dim:词向量维度
        :param num_head:多头注意力中头的数量
        """
        super(Multi_Head_Attention, self).__init__()
        self.device = device
        self.num_head = num_head
        assert embedding_dim % num_head == 0  # 数量关系，这里必须等于0
        self.dim_head = embedding_dim // num_head
        self.fc_Q = nn.Linear(embedding_dim, num_head * self.dim_head).to(device)
        self.fc_K = nn.Linear(embedding_dim, num_head * self.dim_head).to(device)
        self.fc_V = nn.Linear(embedding_dim, num_head * self.dim_head).to(device)
        self.attention = Scaled_Dot_Product_Attention(device)
        self.fc = nn.Linear(num_head * self.dim_head, embedding_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.layer_norm = nn.LayerNorm(embedding_dim).to(device)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out
