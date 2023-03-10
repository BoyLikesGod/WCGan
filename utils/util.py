import logging
import pickle
import json
import os
import time

import torch
import numpy as np


def get_current_time():
    """获取已使用时间"""
    current_time = time.strftime('%Y-%m-%d %H：%M：%S', time.localtime(time.time()))
    return current_time

def get_device(gpu):
    USE_CUDA = torch.cuda.is_available()
    device = torch.device(gpu if USE_CUDA else 'cpu')
    logging.info("GPU: %s, use %s", USE_CUDA, device)
    return device


def load_vocab_and_embedding(dataset):
    embedding_path = f'dataset/{dataset}_embedding.npz'
    vocab_path = f'dataset/{dataset}_vocab.pkl'

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)

    pretrained_embedding = torch.tensor(np.load(embedding_path, allow_pickle=True)['embeddings'].astype("float32"))
    embedding_dim = pretrained_embedding.size(1)

    print('', f'vocab size: {vocab_size}', f'embedding dim: {embedding_dim}', sep='\n-->')

    return vocab, vocab_size, pretrained_embedding

class WrongLabelLogger:
    def __init__(self):
        self.logger = {}
        self.class_accuracy = {}

    def log(self, label_index, out_index, data):
        wrong_index = label_index - out_index
        wrong_index.astype(int)
        for index, label in enumerate(wrong_index):
            if label != 0:
                if data[index] not in self.logger:
                    self.logger[data[index]] = 1
                else:
                    self.logger[data[index]] += 1

    def write(self, path):
        path = path + '/logger.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.logger, f, indent=2, ensure_ascii=False)

def get_scheduler(opt, sc_mode='min', sc_factor=0.8, sc_patience=10, sc_verbose=True, sc_threshold=0.0001,
                  sc_threshold_mode='rel', sc_cooldown=0, sc_min_lr=0, sc_eps=1e-8):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=sc_mode, factor=sc_factor, patience=sc_patience,
                                                           verbose=sc_verbose, threshold=sc_threshold,
                                                           threshold_mode=sc_threshold_mode, cooldown=sc_cooldown,
                                                           min_lr=sc_min_lr, eps=sc_eps)
    return scheduler

def neg_label(label):
    b = torch.ones_like(label)
    label = b + torch.neg(label)

    return label

def get_save_dir():
    current_time = get_current_time()
    save_path = "result/" + current_time
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    return save_path
