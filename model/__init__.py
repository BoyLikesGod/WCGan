import torch
import torch.nn as nn

from utils.util import WrongLabelLogger
from generator import Generator
from discriminator import Discriminator


class Config:
    def __init__(self):
        self.data_numworker = 4  # 加载数据集时的线程

        self.pad_size = 288  # 文本长度 pheme:288
        self.num_class = 2  # 分类的类别数量

        # 词向量
        self.pretrained = None  # 预训练的词向量，会在程序运行时加载
        self.embedding_dim = 100  # 默认embedding的维度

        self.train_batch = 32  # 训练集的batch大小
        self.test_batch = 32  # 测试集的batch大小

        self.epochs = 500  # 训练迭代次数


class WCGan(nn.Module):
    def __init__(self, config):
        super(WCGan, self).__init__()
        self.config = config
        self.mes = f'{config.model_name}  ' \
                   f'dataset: {config.dataset}  ' \
                   f'pad size: {config.pad_size}  ' \
                   f'embedding dim: {config.embedding_dim}  ' \
                   f'dis weight:[{config.ge_weight}, {config.re_weight}]  ' \
                   f'gen weight:[{config.id_weight}, {config.cycle_weight}]'
        print('\n'.join(self.mes.split('  ')))

        self.embedding = nn.Embedding.from_pretrained(config.pretrained, freeze=False).to(self.config.device)

        self.G_r = Generator(self.config)   # 谣言生成器
        self.G_n = Generator(self.config)   # 非谣言生成器
        self.D = Discriminator(self.config) # 判别器

        self.adv_loss = nn.BCEWithLogitsLoss().to(self.config.device)

        # 优化器
        self.G_n_optimizer = torch.optim.RMSprop(self.G_n.parameters(), lr=config.lr_g)
        self.G_r_optimizer = torch.optim.RMSprop(self.G_r.parameters(), lr=config.lr_g)
        self.D_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=config.lr_d)

        self.train_acc, self.test_acc = [], []
        self.train_f1, self.test_f1 = [], []

        self.loss_d, self.loss_gr, self.loss_gn, self.loss_g = 0, 0, 0, 0

        self.wrong_label_logger = WrongLabelLogger()
