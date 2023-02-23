import pickle
import random

import torch
from torch.utils import data
from torch.utils.data.dataset import T_co
import numpy as np


def get_post(content, vocab, pad_size, unk, pad):
    post = []
    for word in content['post'].split(' '):
        idx = vocab.index(word) if word in vocab else unk
        post.append(idx)
    if len(post) < pad_size:
        length = len(post)
        pad_width = pad_size - length
        # 填充pad到post里面，填充的长度为pad_width
        post = np.pad(post, (0, pad_width), mode='constant', constant_values=pad)
    else:
        post = post[:pad_size]
        length = len(post)
    return post, length


def get_label(label):
    mask = torch.zeros(2).float()
    mask[int(label)] = 1.0
    return mask

def data_collate(batch):

    batch.sort(key=lambda data: data[0].shape[0], reverse=True)

    post = [item[0].numpy() for item in batch]
    label = [item[1].numpy() for item in batch]
    length = [item[2] for item in batch]
    name = [item[3] for item in batch]
    news = [item[4] for item in batch]

    post = torch.tensor(np.array(post), dtype=torch.int)
    label = torch.tensor(np.array(label), dtype=torch.float32)

    return post, label, length, name, news

class buildDataset(data.Dataset):
    def __init__(self, config, mode):
        """
        构建数据集
        :param config: 参数
        :param mode:'rumor_train', 'norumor_train', 'train', 'test'
        """
        super(buildDataset, self).__init__()
        self.index = f'dataset/{config.dataset}_index.pkl'
        # 数据集索引，一个字典{'rumor_train':a, 'rumor_test':b, 'norumor_train':c, 'norumor_test':d},a, b, c, d 是四个列表，存放着数据集中推文的id
        self.dataset = f'dataset/{config.dataset}_data.pkl'
        # 数据集，一个字典{post name, label, post, len}

        self.pad_size = config.pad_size  # 句子长度
        self.vocab = config.vocab  # 词表，一个列表，存放着所有的词

        self.unk = config.vocab_size - 2
        self.pad = config.vocab_size - 1

        self.data = []
        with open(self.index, 'rb') as f:
            index = pickle.load(f)[mode]
        with open(self.dataset, 'rb') as f:
            datasets = pickle.load(f)
        """data_index = dataset_index"""
        random.shuffle(index)
        for post in index:
            self.data.append(datasets[post])

    def __getitem__(self, index) -> T_co:
        """
        :param index:
        :return:
            post:torch.Tensor
            label: torch.Tensor non-rumor[1., 0.] rumor[0., 1.]
        """
        content = self.data[index]
        post, length = get_post(content, self.vocab, self.pad_size, self.unk, self.pad)

        # 将句子中的词，转换为词序列
        label = content['label']
        name = content['post name']
        news = content['news']

        post = torch.tensor(post).type(torch.int32)
        label = get_label(label)

        return post, label, length, name, news

    def __len__(self):
        return len(self.data)

def get_dataloader(config, mod):
    current_dataset = buildDataset(config, mod)
    if mod == 'train':
        batch = config.train_batch
    elif mod == 'test':
        batch = config.test_batch
    else:
        batch = config.eval_batch
    current_dataloader = data.DataLoader(current_dataset, batch_size=batch,
                                             num_workers=config.data_numworker, shuffle=True,
                                             collate_fn=data_collate, drop_last=True)
    return current_dataloader

def get_eval_dataloader(config):
    ch_dataloader = get_dataloader(config, 'charlie-hebdo')
    fe_dataloader = get_dataloader(config, 'ferguson')
    ge_dataloader = get_dataloader(config, 'germanwings-crash')
    ot_dataloader = get_dataloader(config, 'ottawa-shooting')
    pu_dataloader = get_dataloader(config, 'putin-missing')
    sy_dataloader = get_dataloader(config, 'sydney-siege')
    test_all = get_dataloader(config, 'test')
    train_all = get_dataloader(config, 'train')

    dataloader = {'train-all': train_all,
                  'test-all': test_all,
                  'charlie-hebdo': ch_dataloader,
                  'ferguson': fe_dataloader,
                  'germanwings-crash': ge_dataloader,
                  'ottawa-shooting': ot_dataloader,
                  'putin-missing': pu_dataloader,
                  'sydney-siege': sy_dataloader}

    return dataloader
