import random

import torch
import numpy as np


class BaseTrainer(object):
    def __init__(self, config, net, optimizer=None):
        self.set_seed(2020)
        self.config = config
        self.net = net
        self.device = "cuda:{}".format(self.config.gpu_id) if torch.cuda.is_available() else "cpu"
        # hyper params
        self.EPOCH = self.config.epoch_num
        self.BATCH_SIZE = self.config.batch_size
        # checkpoint
        self.exp_path = self.config.train_pth_save_dir
        self.net.to(self.device)

        if self.config.resume:  # 从文件中读取模型参数
            self.load_weight()
        self.set_optimizer(optimizer)

    def set_seed(self, seed=2020):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    def load_weight(self):
        try:
            self.net.load_state_dict(torch.load(self.exp_path))
            print('Net Parameters Loaded Successfully!')
        except FileNotFoundError:
            print('Can not find feature.pkl !')

    def set_optimizer(self, optimizer_name):
        if optimizer_name is 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        elif optimizer_name is 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.learning_rate, momentum=0.9,
                                             weight_decay=5e-4)

    def train(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
