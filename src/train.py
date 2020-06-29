import os
import shutil


import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from models.model import network
from utils.parse_yaml import Config
from workers.dbg_trainer import DBGTrainer
from data_loader import MyDataSet


# prepare for training
if not torch.cuda.is_available():
    print('Only train on CPU.')
writer = SummaryWriter(logdir="logs/")
if os.path.exists('logs/'):
    shutil.rmtree('logs/')
config = Config()

# dataset
train_dl = DataLoader(MyDataSet(config, mode='training'), batch_size=config.batch_size,
                      shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
val_dl = DataLoader(MyDataSet(config, mode='validation'), batch_size=config.batch_size,
                    shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

# model
model = network(config)
trainer = DBGTrainer(config, model, train_dl, val_dl, writer=writer)

trainer.run()


