import torch
from model.discriminator import Discriminator
from utils.util import get_device


class Config:
    def __init__(self):
        self.dataset = 'pheme9'
        self.device = get_device('cuda:2')
        self.data_numworker = 4  # 加载数据集时的线程
        self.train_batch = 32  # 训练集的batch大小
        self.test_batch = 64  # 测试集的batch大小
        self.eval_batch = 4
        self.epochs = 500  # 训练迭代次数

        # 词向量
        self.pad_size = 350  # 文本长度 pheme:288
        self.embedding_dim = 100
        self.num_class = 2  # 分类的类别数量

        # gan config
        self.lr_g = 0.001
        self.lr_d = 0.01
        # 训练判别器时，生成数据在判别器上的损失权重
        self.ge_weight = 0.5
        # 训练生成器的损失权重
        self.id_weight = 4
        self.cycle_weight = 8

        # generator config
        self.gen_rnn_hidden_size = self.embedding_dim
        self.gen_rnn_num_layers = 4
        self.gen_rnn_dropout = 0.5

        # discriminator config
        self.dis_rnn_hidden_size = self.embedding_dim  # LSTM中隐藏层的大小
        self.dis_rnn_lstm_layers = 2  # RNN中LSTM的层数
        self.dis_rnn_dropout = 0.5  # LSTM中的dropout rate

        self.dis_cnn_num_filter = 256
        self.dis_cnn_filter_size = [2, 4, 8]
        self.dis_cnn_dropout = 0.4

        # transformer generator
        self.gen_num_encoder = 6
        self.gen_num_head = 4
        self.gen_feed_forward_hidden_size = 256
        self.gen_encoder_dropout = 0.5
        self.gen_positional_dropout = 0.5  # 位置编码时的dropout rate


if __name__ == '__main__':
    config = Config()
    print(1)
    dis = Discriminator(config)
    print(2)
    dis.load_state_dict(torch.load(f'result/2023-03-13 16：12：50/bestAccuracy.pt'))
    print(3)
    print(dis.lstm.all_weights)