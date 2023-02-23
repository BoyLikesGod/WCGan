import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.att = Attention(config.embedding_dim, config.device)

        self.lstm = nn.LSTM(config.embedding_dim * 2,
                            config.dis_rnn_hidden_size,
                            config.dis_rnn_lstm_layers,
                            bias=True,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dis_rnn_dropout).to(config.device)

        self.lstm_fc = nn.Linear(config.dis_rnn_hidden_size * 2, config.dis_rnn_hidden_size).to(config.device)

        self.cnn = nn.ModuleList(
            [nn.Conv2d(1, config.dis_cnn_num_filter, (k, config.dis_rnn_hidden_size)) for k in
             config.dis_cnn_filter_size]).to(config.device)
        self.dropout = nn.Dropout(config.dis_cnn_dropout)
        self.cnn_fc = nn.Linear(config.dis_cnn_num_filter * len(config.dis_cnn_filter_size), config.dis_rnn_hidden_size).to(config.device)

        self.fc = nn.Linear(config.dis_rnn_hidden_size * 2, config.num_class).to(config.device)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, emb, state=None):
        batch, pad_size, embedding_size = emb.size()
        if state is None:
            h = torch.randn(self.config.dis_rnn_lstm_layers * 2, batch, self.config.dis_rnn_hidden_size).float().to(
                self.config.device)
            c = torch.randn(self.config.dis_rnn_lstm_layers * 2, batch, self.config.dis_rnn_hidden_size).float().to(
                self.config.device)
        else:
            h, c = state

        out_emb = self.att(emb)
        out_lstm = torch.cat((out_emb, emb), -1)
        out_lstm, state = self.lstm(out_lstm, (h, c))   # out_lstm: (batch, pad_size, embedding_dim * 2)
        out_lstm = self.lstm_fc(out_lstm[:, -1, :])

        out_cnn = out_emb.unsqueeze(1)
        out_cnn = torch.cat([self.conv_and_pool(out_cnn, conv) for conv in self.cnn], 1)
        out_cnn = self.dropout(out_cnn)
        out_cnn = self.cnn_fc(out_cnn)

        out = torch.cat((out_lstm, out_cnn), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.device = 'cpu'
            self.embedding_dim = 100

            self.dis_rnn_hidden_size = 100
            self.dis_rnn_lstm_layers = 100
            self.dis_rnn_dropout = 0.5

            self.dis_cnn_num_filter = 256
            self.dis_cnn_filter_size = [4, 8, 16]
            self.dis_cnn_dropout = 0.4

            self.num_class = 2


    config = Config()
    dis = Discriminator(config)
    inputs = torch.randn([4, 100, 100])
    print(inputs.size())
    out = dis(inputs)
    print(out.size())
