import time

import torch

import model as Module
from utils.util import load_vocab_and_embedding, predict
from utils.dataset import get_eval_dataloader

if __name__ == '__main__':
    start_time = time.time()

    # load model config
    config = Module.Config()
    print(f"dataset: {config.dataset}", f"pad size: {config.pad_size}", sep='\n-->')

    # load vocab
    print("Loading Vocab...")
    config.vocab, config.vocab_size, config.pretrained = load_vocab_and_embedding(config.dataset)
    config.embedding_dim = config.pretrained.size(1)

    print('get dataloader')
    dataloader = get_eval_dataloader(config)

    print('load model')
    mod = torch.load('result/2023-04-13 20：04：44/bestAccuracy.pth')
    predict(mod, config, dataloader)

    end_time = time.time()
    print(f'End        total run time: {end_time - start_time}')