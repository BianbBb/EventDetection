import torch
from torch.utils.data import DataLoader

from models.model import network
from utils.read_config import Config
from workers.detr_dataloader import MyDataSet, getDatasetDict
from workers.detr_trainer import DetrTrainer


# prepare for training
if not torch.cuda.is_available():
    print('Only train on CPU.')

config = Config()


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


# dataset
train_dict, val_dict, test_dict = getDatasetDict(config.video_info_file, config.video_filter)
dataset_train = MyDataSet(config, train_dict,flag='train')
dataset_val = MyDataSet(config, val_dict,flag='val')

train_dl = DataLoader(dataset_train, config.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=8, pin_memory=True)
val_dl = DataLoader(dataset_val, config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8, pin_memory=True)


model = network(config)

trainer = DetrTrainer(config, model, train_dl, val_dl, optimizer='SGD')
trainer.run()



