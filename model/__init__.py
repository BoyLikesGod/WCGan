from tqdm import tqdm
import os

import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics

from utils.target import draw_target
from utils.util import WrongLabelLogger, get_scheduler, neg_label, get_device
from utils.loss import SinkhornDistance
from model.transformer import Generator
from model.discriminator import Discriminator


class Config:
    def __init__(self):
        self.dataset = 'pheme9'
        self.device = get_device('cuda:1')
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


class WCGan(nn.Module):
    def __init__(self, config):
        super(WCGan, self).__init__()
        self.config = config
        self.mes = f'loss 3 adam rnn dis2 trans gen ' \
                   f'dataset: {config.dataset}  ' \
                   f'pad-emb:[{config.pad_size}, {config.embedding_dim}]  ' \
                   f'dis/gen weight:[{config.ge_weight}] [{config.id_weight}, {config.cycle_weight}]  ' \
                   f'layers:[{config.gen_rnn_num_layers}, {config.dis_rnn_lstm_layers}]'
        print('\n'.join(self.mes.split('  ')))

        self.embedding = nn.Embedding.from_pretrained(config.pretrained, freeze=True).to(self.config.device)

        self.G_r = Generator(self.config)  # 谣言生成器
        self.G_n = Generator(self.config)  # 非谣言生成器
        self.D = Discriminator(self.config)  # 判别器

        self.adv_loss = nn.BCEWithLogitsLoss().to(self.config.device)

        # 优化器
        self.G_n_optimizer = torch.optim.RMSprop(self.G_n.parameters(), lr=config.lr_g)
        self.G_r_optimizer = torch.optim.RMSprop(self.G_r.parameters(), lr=config.lr_g)
        # self.D_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=config.lr_d)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=config.lr_d)

        self.train_acc, self.test_acc = [], []
        self.train_f1, self.test_f1 = [], []

        self.loss_d, self.loss_gr, self.loss_gn, self.loss_g, self.best_acc = 0, 0, 0, 0, 0

        self.wrong_label_logger = WrongLabelLogger()

    def forward(self, norumor_train_dataloader, rumor_train_dataloader, test_dataloader):
        # 加载scheduler
        scheduler_D = get_scheduler(self.D_optimizer)
        scheduler_Gr = get_scheduler(self.G_r_optimizer, sc_patience=8)
        scheduler_Gn = get_scheduler(self.G_n_optimizer, sc_patience=8)

        # 训练
        for epoch in range(self.config.epochs):
            print(f'----------------------------------Epoch : {epoch + 1}----------------------------------')
            loop = tqdm(enumerate(zip(norumor_train_dataloader, rumor_train_dataloader)),
                        total=min(len(norumor_train_dataloader), len(rumor_train_dataloader)))
            loop.set_description(f"Training--[Epoch {epoch + 1} : {self.config.epochs}]")
            predict, true = np.array([]), np.array([])
            flag = torch.randint(0, 2, [1])  # 0:norumor 1:rumor
            # 一个epoch的训练过程
            for i, data in loop:
                # flag+ 1， 上次是谣言，这次就训练非谣言
                flag = (flag + 1) % 2
                # 加载数据
                or_post, or_label = data[flag][0].to(self.config.device), data[flag][1].to(self.config.device)
                or_post = self.embedding(or_post)  # embedding，将词序列转换成词向量

                #  Train Discriminator
                self.train_discriminator(flag, or_post, or_label)
                #  Train Generator
                if i % 3 == 0:
                    self.train_generator(flag, data)

                with torch.no_grad():
                    out = self.D(or_post)
                true_label = torch.argmax(or_label, 1).cpu().numpy()
                out_label = torch.argmax(out, 1).cpu().numpy()
                true = np.append(true, true_label)
                predict = np.append(predict, out_label)

                self.wrong_label_logger.log(true_label, out_label, data[flag][3])

                acc = metrics.accuracy_score(true, predict)
                f1 = metrics.f1_score(true, predict, average=None)
                # f1_s = metrics.f1_score(true, predict, average='weighted')
                f1_s = metrics.f1_score(true, predict, average='macro')

                loop.set_postfix(acc=format(acc, '.5f'), f1=f'[{f1_s}, {f1}]',
                                 loss="[{:.4f}|{:.4f}]".format(self.loss_d, self.loss_g))
            test_acc, test_f1, test_pre, test_recall, test_loss = self.test(test_dataloader, epoch)
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                torch.save(self.D, os.path.join(self.config.save_dir, 'bestAccuracy.pth'))  # 全量保存模型
                print('model saved')
            self.train_acc.append(acc)
            self.test_acc.append(test_acc)
            self.train_f1.append(f1_s)
            self.test_f1.append(test_f1)

            scheduler_D.step(self.loss_d)
            scheduler_Gr.step(self.loss_gr)
            scheduler_Gn.step(self.loss_gn)

        draw_target(self.train_acc, self.test_acc, self.train_f1, self.test_f1, self.config.save_dir, self.mes)
        self.wrong_label_logger.write(self.config.save_dir)

    def train_discriminator(self, flag, or_post, or_label):
        self.D_optimizer.zero_grad()

        or_out = self.D(or_post)
        or_loss = self.adv_loss(or_out, or_label)

        if flag == 0:  # norumor
            ge_post = self.G_r(or_post)
        else:  # rumor
            ge_post = self.G_n(or_post)

        ge_out = self.D(ge_post)
        ge_loss = self.adv_loss(ge_out, neg_label(or_label))

        loss_d = or_loss + ge_loss * self.config.ge_weight
        loss_d.backward(retain_graph=True)

        self.loss_d = loss_d
        self.D_optimizer.step()

    def train_generator(self, flag, data):
        or_post, or_label = data[flag][0].to(self.config.device), data[flag][1].to(self.config.device)
        or_post = self.embedding(or_post)  # embedding，将词序列转换成词向量
        an_post = data[(flag + 1) % 2][0].to(self.config.device)
        an_post = self.embedding(an_post)

        if flag == 0:  # norumor
            self.G_r_optimizer.zero_grad()
            id_post = self.G_r(an_post)
            ge_post = self.G_r(or_post)
            re_post = self.G_n(ge_post)
        else:  # rumor
            self.G_n_optimizer.zero_grad()
            id_post = self.G_n(an_post)
            ge_post = self.G_n(or_post)
            re_post = self.G_r(ge_post)

        # Cycle模型损失，不使用id损失
        """loss_ge = self.adv_loss(self.D(ge_post), neg_label(or_label))
        loss_cycle = -torch.mean(or_post) + torch.mean(re_post)
        loss_g = loss_ge + loss_cycle * self.config.cycle_weight"""

        # Cycle模型损失1
        """loss_ge = self.adv_loss(self.D(ge_post), neg_label(or_label))
        loss_cy = -torch.mean(or_post) + torch.mean(re_post)
        loss_id = -torch.mean(an_post) + torch.mean(id_post)
        loss_g = loss_ge + loss_id * self.config.id_weight + loss_cy * self.config.cycle_weight"""

        # Cycle模型损失2
        """loss_ge = self.adv_loss(self.D(ge_post), neg_label(or_label))
        loss_cy = -torch.mean(or_post) + torch.mean(re_post)
        loss_id = self.adv_loss(self.D(id_post), self.D(an_post))
        loss_g = loss_ge + loss_id * self.config.id_weight + loss_cy * self.config.cycle_weight"""

        # Cycle模型损失3
        # 这个损失的结果是最好的
        loss_ge = -torch.mean(or_post) + torch.mean(ge_post)
        loss_cy = -torch.mean(or_post) + torch.mean(re_post)
        loss_id = self.adv_loss(self.D(id_post), self.D(an_post))
        loss_g = loss_ge + loss_id * self.config.id_weight + loss_cy * self.config.cycle_weight

        # Cycle模型损失3.5
        """loss_ge = -torch.mean(or_post) + torch.mean(ge_post)
        loss_cy = -torch.mean(or_post) + torch.mean(re_post)
        loss_id = -torch.mean(an_post) + torch.mean(id_post)
        loss_g = loss_ge + loss_id * self.config.id_weight + loss_cy * self.config.cycle_weight"""

        # Cycle模型损失4 效果挺好
        """loss_ge = self.adv_loss(self.D(ge_post), neg_label(or_label))
        loss_cy = torch.dist(or_post, re_post, p=1) * 0.001
        loss_id = torch.dist(an_post, id_post, p=1) * 0.001
        loss_g = loss_ge + loss_id * self.config.id_weight + loss_cy * self.config.cycle_weight"""

        # Cycle模型损失5，不使用损失改进
        """loss_ge = self.adv_loss(self.D(ge_post), neg_label(or_label))
        loss_cy = self.adv_loss(self.D(re_post), or_label)
        loss_id = self.adv_loss(self.D(id_post), self.D(an_post))
        loss_g = loss_ge + loss_id * self.config.id_weight + loss_cy * self.config.cycle_weight"""

        # Cycle模型损失6
        """loss_ge = -torch.mean(or_post) + torch.mean(ge_post)
        loss_cy = -torch.mean(or_post) + torch.mean(re_post)
        loss_id = torch.dist(an_post, id_post, p=1) * 0.0001
        loss_g = loss_ge + loss_id * self.config.id_weight + loss_cy * self.config.cycle_weight"""

        # Cycle模型损失7，改用wasserstein_distance计算损失，id损失使用欧几里得距离，没啥用
        """loss_ge = self.wasserstein_loss(or_post, ge_post)[0].to(self.config.device)
        loss_cy = self.wasserstein_loss(or_post, re_post)[0].to(self.config.device)
        loss_id = torch.dist(an_post, id_post, p=1) * 0.001
        loss_g = loss_ge + loss_id * self.config.id_weight + loss_cy * self.config.cycle_weight"""

        # re_out = self.D(re_post)
        # loss_ge = -torch.mean(self.D(or_post)) + torch.mean(self.D(ge_post))
        # loss_id = -torch.mean(self.D(an_post)) + torch.mean(self.D(id_post))
        # loss_id = self.adv_loss(self.D(id_post), neg_label(or_label))
        # loss_cy = -torch.mean(self.D(or_post)) + torch.mean(self.D(re_post))

        # GAN模型损失
        """ge_out = self.D(ge_post)
        loss_ge = self.adv_loss(ge_out, neg_label(or_label))
        re_loss = torch.dist(or_post, re_post, p=1) * 0.001
        loss_g = loss_ge + re_loss"""

        loss_g.backward(retain_graph=True)
        self.loss_g = loss_g

        self.D.zero_grad()
        if flag == 0:
            self.loss_gr = loss_g
            self.G_n.zero_grad()
            self.G_r_optimizer.step()
        else:
            self.loss_gn = loss_g
            self.G_r.zero_grad()
            self.G_n_optimizer.step()

    def test(self, dataloader, epoch):
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        loop.set_description(f"Testing---[Epoch {epoch + 1} : {self.config.epochs}]")
        predicts_all, labels_all = np.array([], dtype=int), np.array([], dtype=int)
        loss_all = 0
        with torch.no_grad():
            for i, data in loop:
                post, label = data[0].to(self.config.device), data[1].to(self.config.device)
                post = self.embedding(post)
                out = self.D(post)
                loss = self.adv_loss(out, label)
                loss_all = loss_all + loss

                labels = torch.argmax(label, 1).data.cpu().numpy()
                predicts = torch.argmax(out.data, 1).cpu().numpy()

                labels_all = np.append(labels_all, labels)
                predicts_all = np.append(predicts_all, predicts)

                self.wrong_label_logger.log(labels, predicts, data[3])

                acc = metrics.accuracy_score(labels_all, predicts_all)
                f1 = metrics.f1_score(labels_all, predicts_all, average=None)
                # f1_s = metrics.f1_score(labels_all, predicts_all, average='weighted')
                f1_s = metrics.f1_score(labels_all, predicts_all, average='macro')
                precision = metrics.precision_score(labels_all, predicts_all, average='macro')
                recall = metrics.recall_score(labels_all, predicts_all, average='macro')
                total_loss = (loss_all / i).item()

                loop.set_postfix(acc=format(acc, '.5f'), f1=f'[{f1_s}, {f1}]', pr=f'[{precision, recall}]',
                                 test_loss=format(total_loss, '.4f'))
        return acc, f1_s, precision, recall, total_loss

    def predict(self, dataloaders):
        for dataloader in dataloaders:
            predicts_all, labels_all = np.array([], dtype=int), np.array([], dtype=int)
            with torch.no_grad():
                for i in range(10):
                    for item, data in tqdm(enumerate(dataloaders[dataloader])):
                        post, label = data[0].to(self.config.device), data[1].to(self.config.device)
                        post = self.embedding(post)
                        out = self.D(post)

                        labels = torch.argmax(label, 1).data.cpu().numpy()
                        predicts = torch.argmax(out.data, 1).cpu().numpy()

                        labels_all = np.append(labels_all, labels)
                        predicts_all = np.append(predicts_all, predicts)

            acc = metrics.accuracy_score(labels_all, predicts_all)
            f1_score = metrics.f1_score(labels_all, predicts_all, average='macro')
            precision = metrics.precision_score(labels_all, predicts_all, average='macro')
            recall = metrics.recall_score(labels_all, predicts_all, average='macro')
            print(f' #################### {dataloader} #################### ')
            print(f'acc: {acc}', f'f1_score: {f1_score}', f'precision: {precision}', f'recall: {recall}', sep='--->')
