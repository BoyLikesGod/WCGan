import logging
import random
import time

import torch
from torch.utils import data
import numpy as np

import model as Module
import utils.util as ut
from utils.dataset import buildDataset, data_collate, get_eval_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set random seed
seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == '__main__':
    start_time = time.time()

    # load model config
    config = Module.Config()
    print(f"dataset: {config.dataset}", f"pad size: {config.pad_size}", sep='\n-->')

    # load vocab
    logging.info("Loading Vocab...")
    config.vocab, config.vocab_size, config.pretrained = ut.load_vocab_and_embedding(config.dataset)
    config.embedding_dim = config.pretrained.size(1)

    # build dataset
    logging.info("Building Dataset...")
    rumor_train_dataset = buildDataset(config, 'rumor_train')
    norumor_train_dataset = buildDataset(config, 'norumor_train')
    test_dataset = buildDataset(config, 'test')

    # load data
    logging.info("Data Loading...")
    rumor_train_dataloader = data.DataLoader(rumor_train_dataset, batch_size=config.train_batch,
                                             num_workers=config.data_numworker, shuffle=True,
                                             collate_fn=data_collate, drop_last=True)
    norumor_train_dataloader = data.DataLoader(norumor_train_dataset, batch_size=config.train_batch,
                                               num_workers=config.data_numworker, shuffle=True,
                                               collate_fn=data_collate, drop_last=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=config.test_batch,
                                      num_workers=config.data_numworker, shuffle=True,
                                      collate_fn=data_collate, drop_last=True)

    # build model
    logging.info("Building Model...")
    config.save_dir = ut.get_save_dir()     # 数据保存地址
    model = Module.WCGan(config)
    print('model', model, sep='------->')
    model(norumor_train_dataloader, rumor_train_dataloader, test_dataloader)

    end_time = time.time()
    logging.info(f'End        total run time: {end_time - start_time}')

    print('eval========>')
    eval_dataloader = get_eval_dataloader(config)
    discriminator = torch.load(f'{config.save_dir}/bestAccuracy.pth')
    ut.predict(discriminator, config, eval_dataloader)
