# 单独训练判别器
import logging
import random
import time

import torch
from sklearn import metrics
from torch import nn
from torch.utils import data
import numpy as np
from tqdm import tqdm

import model as Module
import utils.util as ut
from utils.dataset import buildDataset, data_collate, get_eval_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set random seed
seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class Config:
    def __init__(self):
        self.dataset = 'pheme9'
        self.device = ut.get_device('cuda:1')
        self.data_numworker = 4  # 加载数据集时的线程
        self.train_batch = 32  # 训练集的batch大小
        self.test_batch = 32  # 测试集的batch大小
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
    start_time = time.time()
    logging.info('Training Discriminator')

    # load model config
    config = Config()
    print(f"dataset: {config.dataset}", f"pad size: {config.pad_size}", sep='\n-->')

    # load vocab
    logging.info("Loading Vocab...")
    config.vocab, config.vocab_size, config.pretrained = ut.load_vocab_and_embedding(config.dataset)
    config.embedding_dim = config.pretrained.size(1)

    # build dataset
    logging.info("Building Dataset...")
    train_dataset = buildDataset(config, 'train')
    test_dataset = buildDataset(config, 'test')

    # load data
    logging.info("Data Loading...")
    train_dataloader = data.DataLoader(train_dataset, batch_size=config.train_batch,
                                       num_workers=config.data_numworker, shuffle=True,
                                       collate_fn=data_collate, drop_last=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=config.test_batch,
                                      num_workers=config.data_numworker, shuffle=True,
                                      collate_fn=data_collate, drop_last=True)

    # build model
    logging.info("Building Model...")
    # config.save_dir = ut.get_save_dir()  # 数据保存地址
    model = Module.discriminator.Discriminator(config, dis=True)
    print('model', model, sep='------->')

    adv_loss = nn.BCELoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_d)
    train_acc, train_f1, test_acc, test_f1 = [], [], [], []
    loss_d = []
    best_acc, best_f1 = 0, 0
    for epoch in range(config.epochs):
        model.train()
        loop = tqdm(train_dataloader, total=len(train_dataloader))
        loop.set_description(f"Training--[Epoch {epoch + 1} : {config.epochs}]")
        predict, true = np.array([]), np.array([])
        for x in loop:
            post, label = x[0].to(config.device), x[1].to(config.device)
            out = model(post)
            model.zero_grad()
            loss = adv_loss(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            true_label = torch.argmax(label, 1).cpu().numpy()
            out_label = torch.argmax(out, 1).cpu().numpy()
            true = np.append(true, true_label)
            predict = np.append(predict, out_label)

            acc = metrics.accuracy_score(true, predict)
            f1 = metrics.f1_score(true, predict, average=None)
            f1_s = metrics.f1_score(true, predict, average='macro')
            loop.set_postfix(acc=format(acc, '.5f'), f1=f'[{f1_s}, {f1}]', loss="[{:.4f}]".format(loss))

        model.eval()
        loop = tqdm(test_dataloader, total=len(test_dataloader))
        loop.set_description(f"Testing--[Epoch {epoch + 1} : {config.epochs}]")
        predict, true = np.array([]), np.array([])
        for x in loop:
            with torch.no_grad():
                post, label = x[0].to(config.device), x[1].to(config.device)
                out = model(post)
                loss = adv_loss(out, label)

                true_label = torch.argmax(label, 1).cpu().numpy()
                out_label = torch.argmax(out, 1).cpu().numpy()
                true = np.append(true, true_label)
                predict = np.append(predict, out_label)

                acc = metrics.accuracy_score(true, predict)
                f1 = metrics.f1_score(true, predict, average=None)
                f1_s = metrics.f1_score(true, predict, average='macro')
                loop.set_postfix(acc=format(acc, '.5f'), f1=f'[{f1_s}, {f1}]', loss="[{:.4f}]".format(loss))
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            print(f'Epoch {epoch}: current best acc: {best_acc}, best f1 score: {best_f1}')

    end_time = time.time()
    logging.info(f'End        total run time: {end_time - start_time}')
