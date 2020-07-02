import time
import os
import datetime

import torch
import numpy as np
from runx.logx import logx

from .base_trainer import BaseTrainer
from utils.misc import AverageMeter
from models.detr.matcher import build_matcher
from models.detr.detr import PostProcess
from utils.read_config import Config
from losses import SetCriterion
config = Config()
logx.initialize(logdir=config.log_dir, coolname=True, tensorboard=True)


class DetrTrainer(BaseTrainer):
    def __init__(self, config, net, train_loader, val_loader=None, optimizer=None, writer=None):
        super(DetrTrainer, self).__init__(config, net, optimizer, )
        # log manager

        # dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        # loss
        self.aux_loss = config.aux_loss
        self.criterion = self.init_criterion()
        self.postprocessors = {'bbox': PostProcess()}

        self.BEST_VAL_LOSS = None  # 在验证集上的最好结果
        self.VAL_LOSS = None

        self.net = net.to(self.device)
        self.loss_map = ['labels', 'boxes', 'cardinality']
        self.criterion = self.init_criterion()

    def init_criterion(self):
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        # 根据config进行设置
        # TODO this is a hack
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
        print("Start training")
        start_time = time.time()
        for epoch in range(config.epoch_num):
            # if args.distributed:
            #     sampler_train.set_epoch(epoch)
            torch.cuda.empty_cache()
            logx.msg('|  Train  Epoch : {} ------------------------  |'.format(epoch))
            self.train()
            logx.msg('|  Val  Epoch : {} ------------------------  |'.format(epoch))
            self.val()

            logx.msg('|Val Loss: {:.4f}'.format(np.mean(self.VAL_LOSS)))
            if self.BEST_VAL_LOSS is None:
                self.BEST_VAL_LOSS = np.mean(self.VAL_LOSS)
                self.save_model()
            else:
                if np.mean(self.VAL_LOSS) <= self.BEST_VAL_LOSS:
                    self.BEST_VAL_LOSS = np.mean(self.VAL_LOSS)
                    self.save_model()

                # for evaluation logs

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logx.msg('Training time {}'.format(total_time_str))

    def train(self):
        return self.run_epoch(self.train_loader, is_train=True)

    @torch.no_grad()
    def val(self):
        return self.run_epoch(self.val_loader, is_train=False)

    def run_epoch(self, data_loader, is_train=True, epoch=0):
        if is_train:
            self.net.train()
        else:
            self.net.eval()

        t0 = time.time()  # epoch timer
        t1 = time.time()  # step timer
        step_time = AverageMeter()
        results = {}
        avg_loss_stats = {L: AverageMeter() for L in self.loss_stats}
        for step, batch in enumerate(data_loader):

            torch.cuda.empty_cache()
            for k in batch:
                batch[k] = batch[k].to(device=self.config.device, non_blocking=True)

            output, loss, loss_stats = self.model_with_loss(batch)
            loss = loss.mean()

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            step_time.update(time.time() - t1)
            t1 = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))

            if step % 100 == 0:
                print('| Step: {:<4d} | Time: {:.2f} | Loss: {:.4f} '
                      '| hm loss: {:.4f} | wh loss: {:.4f} '.format(
                    step, step_time.avg, avg_loss_stats['loss'].avg,
                    avg_loss_stats['hm_loss'].avg, avg_loss_stats['wh_loss'].avg, ))

        if not is_train:
            self.VAL_LOSS = avg_loss_stats['loss'].avg
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        print('| Epoch Time: {:.2f} '.format(time.time() - t0))
        return ret, results

    def save_model(self):
        torch.save(self.net.state_dict(),
                   os.path.join(self.exp_path, 'checkpoint-%d.pth' % self.epoch))
        if self.VAL_LOSS < self.BEST_VAL_LOSS:
            self.BEST_VAL_LOSS = self.VAL_LOSS
            torch.save(self.net.state_dict(),
                       os.path.join(self.exp_path, 'checkpoint_best.pth'))
            print("model saved in ", self.exp_path)
