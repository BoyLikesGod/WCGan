import logging
import pickle
import json

import torch
import numpy as np


def get_device(gpu):
    USE_CUDA = torch.cuda.is_available()
    device = torch.device(gpu if USE_CUDA else 'cpu')
    logging.info("GPU: %s, use %s", USE_CUDA, device)
    return device


def load_vocab_and_embedding(dataset):
    embedding_path = f'dataset/{dataset}/{dataset}_embedding.npz'
    vocab_path = f'dataset/{dataset}/{dataset}_vocab.pkl'

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)

    pretrained_embedding = torch.tensor(np.load(embedding_path)['embeddings'].astype("float32"))
    embedding_dim = pretrained_embedding.size(1)

    print('', f'vocab size: {vocab_size}', f'embedding dim: {embedding_dim}', sep='\n-->')

    return vocab, vocab_size, pretrained_embedding

class WrongLabelLogger:
    def __init__(self):
        self.logger = {}

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
