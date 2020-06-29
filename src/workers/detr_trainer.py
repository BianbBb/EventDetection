# loss  # DETR的loss 二分图匹配损失 GIOU loss
# train.py 调用dataloader、定义模型、optimizer，调用xxx_trainer 进行训练

import time
import torch
import numpy as np
import os
from .base_trainer import BaseTrainer
from losses import IoU_loss


class DetrTrainer(BaseTrainer):
    def __init__(self, config, net, train_loader, val_loader=None, optimizer=None):
        super(DetrTrainer, self).__init__(config, net, optimizer, )
        self.exp_dir = config.exp_dir
        self.exp_name = config.exp_name
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.BEST_VAL_LOSS = None  # 在验证集上的最好结果
        self.VAL_LOSS = None

        self.loss = IoU_loss

        self.config = config


    def freeze_layer(self, Freeze_List):
        pass


    def run(self):
        for epoch in range(self.config.EPOCH):
            torch.cuda.empty_cache()
            print('|  Train  Epoch : {} ------------------------  |'.format(epoch))
            self.train()
            print('|  Val  Epoch : {} ------------------------  |'.format(epoch))
            self.val()

            print('|Val Loss: {:.4f}'.format(np.mean(self.VAL_LOSS)))
            if self.BEST_VAL_LOSS is None:
                self.BEST_VAL_LOSS = np.mean(self.VAL_LOSS)
                self.save_model()
            else:
                if np.mean(self.VAL_LOSS) <= self.BEST_VAL_LOSS:
                    self.BEST_VAL_LOSS = np.mean(self.VAL_LOSS)
                    self.save_model()


    def save_model(self):
        pkl_save_name = 'nut2-{}-{:.3f}.pkl'.format(
            time.strftime("%m%d-%H%M", time.localtime()), self.BEST_VAL_LOSS)
        pkl_save_path = os.path.join(self.exp_dir, pkl_save_name)
        torch.save(self.net.state_dict(), pkl_save_path)

    def train(self):
        return self.run_epoch(self.train_loader, is_train=True)

    @torch.no_grad()
    def val(self):
        return self.run_epoch(self.val_loader, is_train=False)

    def run_epoch(self, data_loader, is_train=True, epoch=0 ):
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
                           avg_loss_stats['hm_loss'].avg,avg_loss_stats['wh_loss'].avg,))

        if not is_train:
            self.VAL_LOSS = avg_loss_stats['loss'].avg
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        print('| Epoch Time: {:.2f} '.format(time.time()-t0))
        return ret, results
