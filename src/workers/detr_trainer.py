import time
import os
import torch
import math
import numpy as np
from runx.logx import logx
from torch.optim.lr_scheduler import LambdaLR

from .base_trainer import BaseTrainer
from models.detr.matcher import build_matcher
from losses import SetCriterion


class DetrTrainer(BaseTrainer):
    def __init__(self, config, net, train_loader, val_loader=None, optimizer='AdamW'):
        super(DetrTrainer, self).__init__(config, net, optimizer)
        # dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        # loss
        self.VAL_LOSS = np.inf
        self.aux_loss = config.aux_loss
        # log manager
        self.logx = logx
        self.logx.initialize(logdir=config.log_dir, coolname=True, tensorboard=True)
        # hyper param
        self.save_config(config.config, os.path.join(config.log_dir, 'hyp.yaml'))
        self.epoch = 0
        self.loss_map = ['classes', 'cardinality', 'segments', ]
        self.criterion = self.init_criterion()
        lf = lambda x: (((1 + math.cos(x * math.pi / config.epoch_num)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lf) if config.lr_schduler else None
        self.max_norm = 0.1

        self.warm_up = [1, 10, 100, 100, 100,100,100,10,10,10,10,10,1]
        # self.warm_up = [1]

    def init_criterion(self):
        weight_dict = {'loss_class': self.config.set_cost_classes, 'loss_segments': self.config.set_cost_segments, 'loss_diou': self.config.set_cost_diou}
        # this is a hack
        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(self.config.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        matcher = build_matcher(self.config)
        criterion = SetCriterion(self.config.num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=self.config.eos_coef, losses=self.loss_map)
        criterion.to(self.device)
        return criterion

    def run(self):
        self.logx.msg("| Start Training")
        print("log saved in", self.config.log_dir)
        for epoch in range(self.EPOCH):
            self.epoch = epoch
            torch.cuda.empty_cache()
            self.logx.msg('| ------------  Train  Epoch : {:<3d} ------------ |'.format(epoch))
            self.train()
            self.logx.msg('| ------------   Val   Epoch : {:<3d} ------------ |'.format(epoch))
            self.val()

    def train(self):
        self.run_epoch(self.train_loader, training=True)

    @torch.no_grad()
    def val(self):
        self.run_epoch(self.val_loader, training=False)

    def run_epoch(self, data_loader, training=True):
        if training:
            self.net.train()
        else:
            self.net.eval()

        epoch_loss_dict = {'total': 0, 'loss_class': 0, 'loss_segments': 0, 'loss_diou': 0}
        epoch_time = time.time()

        warm_up = self.warm_up[self.epoch] if self.epoch < len(self.warm_up) else 1
        warm_up = torch.tensor(warm_up, requires_grad=False).to(self.device)

        for n_iter, (samples, targets,_) in enumerate(data_loader):
            # data
            samples = torch.cat([i.unsqueeze(0) for i in samples], dim=0)
            samples = samples.to(self.device)
            targets = list(targets)
            targets = [{k: torch.FloatTensor(v).to(self.device) for k, v in t.items()} for t in targets]

            # forward
            # print(samples.size())
            outputs = self.net(samples) # samples：（b,c,T）

            # loss
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            step_loss_dict = {'total': 0, 'loss_class': 0, 'loss_segments': 0, 'loss_diou': 0}  # N_step个step的loss
            # step total loss
            step_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # get one step loss
            for k in step_loss_dict.keys():
                if k == 'total':
                    step_loss_dict[k] += step_loss
                else:
                    step_loss_dict[k] += loss_dict[k]
            # cumsum epoch loss
            for k in epoch_loss_dict.keys():
                epoch_loss_dict[k] += step_loss_dict[k]

            step_loss = warm_up * step_loss

            if training:
                self.optimizer.zero_grad()
                step_loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step(self.VAL_LOSS)

            # 每隔N_step打印一次
            N_step = 200
            if (n_iter + 1) % N_step == 0:
                print('| Epoch {:<3d} Step {:<5d} '
                      '| Total Loss: {:5.4f} '
                      '| Class Loss: {:.4f} '
                      '| L1 Loss: {:.4f} '
                      '| DIoU Loss: {:.4f} |'
                      .format(self.epoch, n_iter + 1, step_loss_dict['total'], step_loss_dict['loss_class'],
                              step_loss_dict['loss_segments'], step_loss_dict['loss_diou']
                              ))

        metrics = {
            'total_loss': epoch_loss_dict['total'] / (n_iter+1),
            'ce_loss': epoch_loss_dict['loss_class'] / (n_iter+1),
            'segments_loss': epoch_loss_dict['loss_segments'] / (n_iter+1),
            'iou_loss': epoch_loss_dict['loss_diou'] / (n_iter+1)
        }
        if training:
            self.logx.metric('train', metrics, self.epoch)
        else:
            self.logx.metric('val', metrics, self.epoch)
            self.VAL_LOSS = epoch_loss_dict['total'] / (n_iter + 1)

        self.logx.msg('| Epoch {:<14d} | Total Loss: {:5.4f} | Time: {:<11.0f}s | '
                      .format(self.epoch, epoch_loss_dict['total'] / (n_iter+1), (time.time() - epoch_time)))

        save_dict = {
            'epoch': self.epoch + 1,
            'state_dict': self.net.state_dict()
        }
        self.logx.save_model(save_dict, metric=self.VAL_LOSS, epoch=self.epoch, delete_old=True, higher_better=False)
