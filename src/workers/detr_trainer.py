import torch
import numpy as np
from runx.logx import logx

from .base_trainer import BaseTrainer
from utils.misc import AverageMeter
from models.detr.matcher import build_matcher
from models.detr.detr import PostProcess
from losses import SetCriterion


class DetrTrainer(BaseTrainer):
    def __init__(self, config, net, train_loader, val_loader=None, optimizer=None, writer=None):
        super(DetrTrainer, self).__init__(config, net, optimizer, )
        # dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        # loss
        self.aux_loss = config.aux_loss

        self.postprocessors = {'bbox': PostProcess()}

        self.BEST_VAL_LOSS = None  # 在验证集上的最好结果
        self.VAL_LOSS = None
        # log manager
        self.logx = logx
        self.logx.initialize(logdir=config.log_dir, coolname=True, tensorboard=True)
        self.epoch = 0
        self.loss_map = ['classes', 'cardinality', 'segments', ]
        self.criterion = self.init_criterion()

    def init_criterion(self):
        weight_dict = {'loss_ce': 1, 'loss_segments': 5, 'loss_diou': 2}
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
        self.logx.msg("Start training")
        for epoch in range(self.EPOCH):
            self.epoch = epoch
            torch.cuda.empty_cache()
            logx.msg('|  Train  Epoch : {} ------------------------  |'.format(epoch))
            self.train()
            logx.msg('|  Val  Epoch : {} ------------------------  |'.format(epoch))
            self.val()

            logx.msg('|Val Loss: {:.4f}'.format(np.mean(self.VAL_LOSS)))
            if self.BEST_VAL_LOSS is None:
                self.BEST_VAL_LOSS = np.mean(self.VAL_LOSS)
            else:
                if np.mean(self.VAL_LOSS) <= self.BEST_VAL_LOSS:
                    self.BEST_VAL_LOSS = np.mean(self.VAL_LOSS)

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

        step_time = AverageMeter()
        results = {}
        # avg_loss_stats = {L: AverageMeter() for L in self.loss_stats}
        cost_val = 0
        for n_iter, (samples, targets) in enumerate(data_loader):
            torch.cuda.empty_cache()

            samples = samples.to(self.device)
            outputs = self.net(samples)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            cost_val += losses

            if training:
                self.optimizer.zero_grad()
                losses.backward()
                # TODO: max_norm 的作用？
                # if max_norm > 0:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                self.optimizer.step()
        cost_val /= (n_iter + 1)
        metrics = {
            'total_loss': cost_val
        }
        if training:
            self.logx.metric('train', metrics, self.epoch)
            self.logx.msg(
                "Epoch-%d Training Loss: "
                "Total - %.05f"
                % (self.epoch, cost_val))
        else:
            self.logx.metric('val', metrics, self.epoch)
            self.VAL_LOSS = cost_val
            self.logx.msg(
                "Epoch-%d Validation Loss: "
                "Total - %.05f"
                % (self.epoch, cost_val))

        save_dict = {
            'epoch': self.epoch + 1,
            'state_dict': self.net.state_dict()
        }
        self.logx.save_model(save_dict, metric=self.VAL_LOSS, epoch=self.epoch, delete_old=True)

