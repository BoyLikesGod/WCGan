import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, config, dis=False):
        super(Discriminator, self).__init__()
        self.dis = dis
        self.config = config
        if dis:
            self.embedding = nn.Embedding.from_pretrained(config.pretrained, freeze=True).to(config.device)
        self.att = Attention(config.embedding_dim, config.device)

        self.lstm = nn.LSTM(config.embedding_dim,
                            config.dis_rnn_hidden_size,
                            config.dis_rnn_lstm_layers,
                            bias=True,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dis_rnn_dropout).to(config.device)

        self.fc = nn.Linear(config.dis_rnn_hidden_size * 3, config.num_class).to(config.device)

        if dis:
            self.sigmoid = nn.Sigmoid()

    def forward(self, emb, state=None):
        if self.dis:
            emb = self.embedding(emb)
        batch, pad_size, embedding_size = emb.size()
        if state is None:
            h = torch.randn(self.config.dis_rnn_lstm_layers * 2, batch, self.config.dis_rnn_hidden_size).float().to(
                self.config.device)
            c = torch.randn(self.config.dis_rnn_lstm_layers * 2, batch, self.config.dis_rnn_hidden_size).float().to(
                self.config.device)
        else:
            h, c = state

        """ 判别器架构1
        out = self.att(emb)
        out = torch.cat((out, emb), -1)
        out, state = self.lstm(out, (h, c))
        out = self.fc(out[:, -1, :])"""

        # 判别器架构2
        out, state = self.lstm(emb, (h, c))
        out = self.att(out)
        out = torch.cat((out, emb), -1)
        out = self.fc(out[:, -1, :])
        if self.dis:
            out = self.sigmoid(out)

        return out


class Attention(nn.Module):
    def __init__(self, embedding_dim, device):
        super(Attention, self).__init__()
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.randn(embedding_dim*2, requires_grad=True)).to(device)
        self.w.data.normal_(mean=0.0, std=0.05)

    def forward(self, x):
        M = self.tanh(x)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = x * alpha
        return out
