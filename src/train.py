import torch
from torch.utils.data import DataLoader

from models.model import network
from utils.read_config import Config
from workers.detr_dataloader import MyDataSet

# prepare for training
if not torch.cuda.is_available():
    print('Only train on CPU.')

config = Config()

# dataset
train_dl = DataLoader(MyDataSet(config, mode='training'), batch_size=config.batch_size,
                      shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
# val_dl = DataLoader(MyDataSet(config, mode='validation'), batch_size=config.batch_size,
#                     shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

# model
model = network(config)
# trainer = DetrTrainer(config, model, train_dl, val_dl)
# #
# # trainer.run()

if __name__ == '__main__':
    # from torchsummary import summary
    # summary(model.cuda(),(100,1024),batch_size=2)
    for n_iter, (gt_action, gt_start, gt_end, feature) in enumerate(train_dl):
        torch.cuda.empty_cache()
        print('-----')
        print(gt_action)
        print(gt_start)
        print(gt_end)
        #print(feature)
        print(feature.size())


