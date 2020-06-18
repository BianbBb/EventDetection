import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from model import DBG
from data_loader import MyDataSet
from utils.util import gen_mask
from utils.parse_yaml import Config
from losses import binary_logistic_loss, IoU_loss

config = Config()

checkpoint_dir = config.checkpoint_dir
batch_size = config.batch_size
learning_rate = config.learning_rate
tscale = config.tscale
feature_dim = config.feature_dim
epoch_num = config.epoch_num

""" Initialize map mask
"""
mask = gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), 1)
mask = torch.from_numpy(mask).float().requires_grad_(False).cuda()
tmp_mask = mask.repeat(batch_size, 1, 1, 1).requires_grad_(False)
tmp_mask = tmp_mask > 0


def train(net, dl_iter, optimizer, epoch, training):
    """
    One epoch of runing DBG model
    :param net: DBG network module
    :param dl_iter: data loader
    :param optimizer: optimizer module
    :param epoch: current epoch number
    :param training: bool, training or not
    :return: None
    """
    if training:
        net.train()
    else:
        net.eval()
    loss_action_val = 0
    loss_iou_val = 0
    loss_start_val = 0
    loss_end_val = 0
    cost_val = 0
    for n_iter, \
        (gt_action, gt_start, gt_end, feature, iou_label) in tqdm.tqdm(enumerate(dl_iter)):
        gt_action = gt_action.cuda()
        gt_start = gt_start.cuda()
        gt_end = gt_end.cuda()
        feature = feature.cuda()
        iou_label = iou_label.cuda()

        output_dict = net(feature)
        x1 = output_dict['x1']
        x2 = output_dict['x2']
        x3 = output_dict['x3']
        iou = output_dict['iou']
        prop_start = output_dict['prop_start']
        prop_end = output_dict['prop_end']

        # calculate action loss
        loss_action = binary_logistic_loss(gt_action, x1) + \
                      binary_logistic_loss(gt_action, x2) + \
                      binary_logistic_loss(gt_action, x3)
        loss_action /= 3.0

        # calculate IoU loss
        iou_losses = 0.0
        for i in range(batch_size):
            iou_loss = IoU_loss(iou_label[i:i + 1], iou[i:i + 1])
            iou_losses += iou_loss
        loss_iou = iou_losses / batch_size * 10.0

        # calculate starting and ending map loss
        gt_start = torch.unsqueeze(gt_start, 3).repeat(1, 1, 1, tscale)
        gt_end = torch.unsqueeze(gt_end, 2).repeat(1, 1, tscale, 1)
        loss_start = binary_logistic_loss(
            torch.masked_select(gt_start, tmp_mask),
            torch.masked_select(prop_start, tmp_mask)
        )
        loss_end = binary_logistic_loss(
            torch.masked_select(gt_end, tmp_mask),
            torch.masked_select(prop_end, tmp_mask)
        )

        # total loss
        cost = 2.0 * loss_action + loss_iou + loss_start + loss_end

        if training:
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

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

    if training:
        print(
            "Epoch-%d Train Loss: "
            "Total - %.05f, Action - %.05f, Start - %.05f, End - %.05f, IoU - %.05f"
            % (epoch, cost_val, loss_action_val, loss_start_val, loss_end_val, loss_iou_val))
    else:
        print(
            "Epoch-%d Validation Loss: "
            "Total - %.05f, Action - %.05f, Start - %.05f, End - %.05f, IoU - %.05f"
            % (epoch, cost_val, loss_action_val, loss_start_val, loss_end_val, loss_iou_val))

        torch.save(net.module.state_dict(),
                   os.path.join(checkpoint_dir, 'DBG_checkpoint-%d.ckpt' % epoch))
        if cost_val < net.module.best_loss:
            net.module.best_loss = cost_val
            torch.save(net.module.state_dict(),
                       os.path.join(checkpoint_dir, 'DBG_checkpoint_best.ckpt'))


def set_seed(seed):
    """
    Set randon seed for pytorch
    :param seed:
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('Only train on GPU.')
        exit()
    torch.backends.cudnn.enabled = False # set False to speed up Conv3D operation
    set_seed(2020)
    net = DBG(feature_dim)
    net = nn.DataParallel(net, device_ids=[0]).cuda()

    # set weight decay for different parameters
    Net_bias = []
    for name, p in net.module.named_parameters():
        if 'bias' in name:
            Net_bias.append(p)

    DSBNet_weight = []
    for name, p in net.module.DSBNet.named_parameters():
        if 'bias' not in name:
            DSBNet_weight.append(p)

    PFG_weight = []
    for name, p in net.module.PropFeatGen.named_parameters():
        if 'bias' not in name:
            PFG_weight.append(p)

    ACR_TBC_weight = []
    for name, p in net.module.ACRNet.named_parameters():
        if 'bias' not in name:
            ACR_TBC_weight.append(p)
    for name, p in net.module.TBCNet.named_parameters():
        if 'bias' not in name:
            ACR_TBC_weight.append(p)

    # setup Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': Net_bias, 'weight_decay': 0},
        {'params': DSBNet_weight, 'weight_decay': 2e-3},
        {'params': PFG_weight, 'weight_decay': 2e-4},
        {'params': ACR_TBC_weight, 'weight_decay': 2e-5}
    ], lr=1.0)

    # setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: learning_rate[x])
    # setup training and validation data loader
    train_dl = DataLoader(MyDataSet(config, mode='training'), batch_size=batch_size,
                          shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    val_dl = DataLoader(MyDataSet(config, mode='validation'), batch_size=batch_size,
                        shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    # train DBG
    for i in range(epoch_num):
        print('current learning rate:', scheduler.get_last_lr()[0])
        train(net, train_dl, optimizer, i, training=True)
        train(net, val_dl, optimizer, i, training=False)
        scheduler.step()