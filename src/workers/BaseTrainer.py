import torch
import os


class BaseTrainer(object):
    def __init__(self, config, net, optimizer=None, loss=None, ):
        self.config = config
        self.EPOCH = self.config.EPOCH
        self.BATCH_SIZE = self.config.BATCH_SIZE
        self.net = net
        self.device = self.config.device

        # set seed
        torch.cuda.manual_seed(self.config.SEED)
        # torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True

        self.exp_dir = config.exp_dir
        self.exp_name = config.exp_name  # +'best_checkpoin.pth'
        self.exp_path = os.path.join(self.exp_dir, self.exp_name)

        # if len(self.para.gpu_ids) > 0:
        #     self.net = nn.DataParallel(net, device_ids=self.para.gpu_ids)
        self.net.to(self.device)

        if self.config.resume:  # 从文件中读取模型参数
            self.load_weight()
        self.optimizer_name = optimizer
        self.set_optimizer()

    def load_weight(self):
        try:
            self.net.load_state_dict(torch.load(self.exp_path))
            print('Net Parameters Loaded Successfully!')
        except FileNotFoundError:
            print('Can not find feature.pkl !')

    def set_optimizer(self):
        if self.optimizer_name is 'Adam':
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.config.lr, )

        elif self.optimizer_name is 'SGD':
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay)

    def train(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError

    def get_losses(self):
        raise NotImplementedError

    def run_epoch(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
