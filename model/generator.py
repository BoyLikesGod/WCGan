import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(config.embedding_dim,
                            config.gen_rnn_hidden_size,
                            config.gen_rnn_num_layers,
                            bias=True,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.gen_rnn_dropout).to(self.config.device)

        self.fc = nn.Linear(config.gen_rnn_hidden_size * 2, config.embedding_dim).to(self.config.device)

    def forward(self, x, state=None):
        batch, _, _ = x.size()
        if state is None:
            h = torch.randn(self.config.gen_rnn_num_layers * 2, batch, self.config.gen_rnn_hidden_size).float().to(self.config.device)
            c = torch.randn(self.config.gen_rnn_num_layers * 2, batch, self.config.gen_rnn_hidden_size).float().to(
                self.config.device)
        else:
            h, c = state

        out, state = self.lstm(x, (h, c))
        out = self.fc(out)
        return out