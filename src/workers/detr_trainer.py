import time
import os
import shutil

import torch
import numpy as np
from runx.logx import logx

from .base_trainer import BaseTrainer
from models.detr.matcher import build_matcher
from models.detr.detr import PostProcess
from losses import SetCriterion


class DetrTrainer(BaseTrainer):
    def __init__(self, config, net, train_loader, val_loader=None, optimizer='AdamW'):
        super(DetrTrainer, self).__init__(config, net, optimizer)
        # dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        # loss
        self.aux_loss = config.aux_loss
        self.postprocessors = {'bbox': PostProcess()}  ##############???????????????
        self.VAL_LOSS = np.inf
        # log manager
        self.logx = logx
        if os.path.exists(config.log_dir):
            shutil.rmtree(config.log_dir)
        self.logx.initialize(logdir=config.log_dir, coolname=True, tensorboard=True)
        self.epoch = 0
        self.loss_map = ['classes', 'cardinality', 'segments', ]
        self.criterion = self.init_criterion()
        self.max_norm = 0.1

    def init_criterion(self):
        weight_dict = {'loss_ce': 1, 'loss_segments': 3, 'loss_diou': 2}
        # 根据config进行设置
        # TODO this is a hack
        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(self.config.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        matcher = build_matcher(self.config)
        criterion = SetCriterion(self.config.num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=0.1, losses=self.loss_map)
        criterion.to(self.device)
        return criterion

    def run(self):
        self.logx.msg("| Start Training")
        for epoch in range(self.EPOCH):
            self.epoch = epoch
            torch.cuda.empty_cache()
            self.logx.msg(
                '| ---------------------------------------  Train  Epoch : {} --------------------------------------- |'.format(
                    epoch))
            self.train()
            self.logx.msg(
                '| ---------------------------------------   Val   Epoch : {} --------------------------------------- |'.format(
                    epoch))
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

        epoch_loss_dict = {'total': 0, 'loss_ce': 0, 'loss_segments': 0, 'loss_diou': 0}
        epoch_time = time.time()

        for n_iter, (samples, targets) in enumerate(data_loader):
            # data
            samples = torch.cat([i.unsqueeze(0) for i in samples], dim=0)
            samples = samples.to(self.device)
            targets = list(targets)
            targets = [{k: torch.FloatTensor(v).to(self.device) for k, v in t.items()} for t in targets]

            # forward
            outputs = self.net(samples)

            # loss
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            step_loss_dict = {'total': 0, 'loss_ce': 0, 'loss_segments': 0, 'loss_diou': 0}  # N_step个step的loss
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

            if training:
                self.optimizer.zero_grad()
                step_loss.backward()
                # TODO: max_norm 的作用？
                if self.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm)
                self.optimizer.step()

            # 每隔N_step打印一次
            N_step = 50
            if (n_iter + 1) % N_step == 0:
                # TODO: 累积ce、l1、diou loss step的平均时间
                print('| Epoch {:<3d} Step {:<5d} '
                      '| Total Loss: {:.4f} '
                      '| CE Loss: {:.4f} '
                      '| L1 Loss: {:.4f} '
                      '| DIoU Loss: {:.4f} |'
                      .format(self.epoch, n_iter + 1, step_loss_dict['total'], step_loss_dict['loss_ce'],
                              step_loss_dict['loss_segments'], step_loss_dict['loss_diou']
                              ))

        metrics = {
            'total_loss': epoch_loss_dict['total'] / n_iter+1,
            'ce_loss': epoch_loss_dict['loss_ce'] / n_iter+1,
            'segments_loss': epoch_loss_dict['loss_segments'] / n_iter+1,
            'iou_loss': epoch_loss_dict['loss_diou'] / n_iter+1
        }
        if training:
            self.logx.metric('train', metrics, self.epoch)
        else:
            self.logx.metric('val', metrics, self.epoch)
            self.VAL_LOSS = epoch_loss_dict['total']

        self.logx.msg('| Epoch {:<14d} | Total Loss: {:.4f} | Time: {:<8.0f}s | '
                      .format(self.epoch, epoch_loss_dict['total'], (time.time() - epoch_time)))

        save_dict = {
            'epoch': self.epoch + 1,
            'state_dict': self.net.state_dict()
        }
        self.logx.save_model(save_dict, metric=self.VAL_LOSS, epoch=self.epoch, delete_old=True)
