import sys
import os
import warnings

import torch
import numpy as np
from tqdm import tqdm
from runx.logx import logx

from .base_trainer import BaseTrainer

warnings.filterwarnings('ignore')
sys.path.append("..")
from utils.utils import gen_mask
from losses import binary_logistic_loss, IoU_loss


class DBGTrainer(BaseTrainer):
    def __init__(self, config, net, train_loader, val_loader=None, optimizer=None):
        super(DBGTrainer, self).__init__(config, net, optimizer)
        # dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        # model
        self.mask, self.tmp_mask = self.init_mask()
        self.optimizer = self.init_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.epoch = 0
        # loss
        self.BEST_VAL_LOSS = None  # 在验证集上的最好结果
        self.VAL_LOSS = None
        # log
        self.logx = logx
        self.logx.initialize(logdir=config.log_dir, coolname=True, tensorboard=True)

    def init_mask(self):
        mask = gen_mask(self.config.tscale)
        mask = np.expand_dims(np.expand_dims(mask, 0), 1)
        mask = torch.from_numpy(mask).float().requires_grad_(False).cuda()
        tmp_mask = mask.repeat(self.BATCH_SIZE, 1, 1, 1).requires_grad_(False)
        tmp_mask = tmp_mask > 0
        return mask, tmp_mask

    def init_optimizer(self):
        Net_bias = []
        for name, p in self.net.named_parameters():
            if 'bias' in name:
                Net_bias.append(p)

        DSBNet_weight = []
        for name, p in self.net.DSBNet.named_parameters():
            if 'bias' not in name:
                DSBNet_weight.append(p)

        PFG_weight = []
        for name, p in self.net.PropFeatGen.named_parameters():
            if 'bias' not in name:
                PFG_weight.append(p)

        ACR_TBC_weight = []
        for name, p in self.net.ACRNet.named_parameters():
            if 'bias' not in name:
                ACR_TBC_weight.append(p)
        for name, p in self.net.TBCNet.named_parameters():
            if 'bias' not in name:
                ACR_TBC_weight.append(p)

        # setup Adam optimizer
        optimizer_ = torch.optim.Adam([
            {'params': Net_bias, 'weight_decay': 0},
            {'params': DSBNet_weight, 'weight_decay': 2e-3},
            {'params': PFG_weight, 'weight_decay': 2e-4},
            {'params': ACR_TBC_weight, 'weight_decay': 2e-5}
        ], lr=3e-4)
        return optimizer_

    def run(self):
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        # train DBG
        for i in range(self.EPOCH):
            self.epoch = i
            self.train()
            self.val()
            self.scheduler.step(i)

    def train(self):
        return self.run_epoch(self.train_loader, training=True)

    @torch.no_grad()
    def val(self):
        return self.run_epoch(self.val_loader, training=False)

    def run_epoch(self, data_loader, training):
        if training:
            self.net.train()
        else:
            self.net.eval()
        loss_action_val = 0
        loss_iou_val = 0
        loss_start_val = 0
        loss_end_val = 0
        cost_val = 0
        for n_iter, (gt_action, gt_start, gt_end, feature, iou_label) in tqdm(enumerate(data_loader)):
            gt_action = gt_action.to(device=self.device, non_blocking=True)
            gt_start = gt_start.to(device=self.device, non_blocking=True)
            gt_end = gt_end.to(device=self.device, non_blocking=True)
            feature = feature.to(device=self.device, non_blocking=True)
            iou_label = iou_label.to(device=self.device, non_blocking=True)

            output_dict = self.net(feature)
            x1 = output_dict['x1']
            x2 = output_dict['x2']
            x3 = output_dict['x3']
            iou = output_dict['iou']
            prop_start = output_dict['prop_start']
            prop_end = output_dict['prop_end']

            # calculate action loss
            loss_action = binary_logistic_loss(gt_action, x1) \
                          + binary_logistic_loss(gt_action, x2) \
                          + binary_logistic_loss(gt_action, x3)
            loss_action /= 3.0

            # calculate IoU loss
            iou_losses = 0.0
            for i in range(self.BATCH_SIZE):
                iou_loss = IoU_loss(iou_label[i:i + 1], iou[i:i + 1], self.mask)
                iou_losses += iou_loss
            loss_iou = iou_losses / self.BATCH_SIZE * 10.0

            # calculate starting and ending map loss
            gt_start = torch.unsqueeze(gt_start, 3).repeat(1, 1, 1, self.config.tscale)
            gt_end = torch.unsqueeze(gt_end, 2).repeat(1, 1, self.config.tscale, 1)
            loss_start = binary_logistic_loss(
                torch.masked_select(gt_start, self.tmp_mask),
                torch.masked_select(prop_start, self.tmp_mask)
            )
            loss_end = binary_logistic_loss(
                torch.masked_select(gt_end, self.tmp_mask),
                torch.masked_select(prop_end, self.tmp_mask)
            )

            # total loss
            cost = 2.0 * loss_action + loss_iou + loss_start + loss_end
            if training:
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

            loss_action_val += loss_action.cpu().detach().numpy()
            loss_iou_val += loss_iou.cpu().detach().numpy()
            loss_start_val += loss_start.cpu().detach().numpy()
            loss_end_val += loss_end.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()

        loss_action_val /= (n_iter + 1)
        loss_iou_val /= (n_iter + 1)
        loss_start_val /= (n_iter + 1)
        loss_end_val /= (n_iter + 1)
        cost_val /= (n_iter + 1)

        metrics = {
            'total_loss': cost_val,
            'action_loss': loss_action_val,
            'start_loss': loss_start_val,
            'end_loss': loss_end_val,
            'iou_loss': loss_iou_val,
        }

        if training:
            self.logx.metric('train', metrics, self.epoch)
            self.logx.msg(
                "Epoch-%d Train      Loss: "
                "Total - %.05f, Action - %.05f, Start - %.05f, End - %.05f, IoU - %.05f"
                % (self.epoch, cost_val, loss_action_val, loss_start_val, loss_end_val, loss_iou_val))
        else:
            self.logx.metric('val', metrics, self.epoch)
            self.VAL_LOSS = cost_val
            self.logx.msg(
                "Epoch-%d Validation Loss: "
                "Total - %.05f, Action - %.05f, Start - %.05f, End - %.05f, IoU - %.05f"
                % (self.epoch, cost_val, loss_action_val, loss_start_val, loss_end_val, loss_iou_val))
        save_dict = {
            'epoch': self.epoch+1,
            'state_dict': self.net.state_dict()
        }
        self.logx.save_model(save_dict, metric=self.VAL_LOSS, epoch=self.epoch, delete_old=False)
