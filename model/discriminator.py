import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.embedding = nn.Embedding.from_pretrained(config.pretrained, freeze=False).to(config.device)
        self.att = Attention(config.embedding_dim, config.device)

        self.lstm = nn.LSTM(config.embedding_dim * 2,
                            config.dis_rnn_hidden_size,
                            config.dis_rnn_lstm_layers,
                            bias=True,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dis_rnn_dropout).to(config.device)

        self.fc = nn.Linear(config.dis_rnn_hidden_size * 2, config.num_class).to(config.device)

    def forward(self, x, state=None):
        emb = self.embedding(x)
        batch, pad_size, embedding_size = emb.size()
        if state is None:
            h = torch.randn(self.config.dis_rnn_lstm_layers * 2, batch, self.config.dis_rnn_hidden_size).float().to(
                self.config.device)
            c = torch.randn(self.config.dis_rnn_lstm_layers * 2, batch, self.config.dis_rnn_hidden_size).float().to(
                self.config.device)
        else:
            h, c = state

        out = self.att(emb)
        out = torch.cat((out, emb), -1)
        out, state = self.lstm(out, (h, c))
        out = self.fc(out[:, -1, :])
        return out


class Attention(nn.Module):
    def __init__(self, embedding_dim, device):
        super(Attention, self).__init__()
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.randn(embedding_dim, requires_grad=True)).to(device)
        self.w.data.normal_(mean=0.0, std=0.05)

    def forward(self, x):
        M = self.tanh(x)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = x * alpha
        return out
