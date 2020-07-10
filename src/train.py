import torch
from torch.utils.data import DataLoader

from models.model import network
from utils.read_config import Config
from workers.detr_dataloader import MyDataSet,getDatasetDict
from workers.detr_trainer import DetrTrainer
import time


# prepare for training
if not torch.cuda.is_available():
    print('Only train on CPU.')

config = Config()


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


#start_time = time.time()
# dataset
train_dict, val_dict, test_dict = getDatasetDict(config, config.video_info_file, config.video_filter)
dataset_train = MyDataSet(config, train_dict)
dataset_val = MyDataSet(config, val_dict)

# if args.distributed:
#     sampler_train = DistributedSampler(dataset_train)
#     sampler_val = DistributedSampler(dataset_val, shuffle=False)
# else:
sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, config.batch_size, drop_last=True)

train_dl = DataLoader(dataset_train, batch_sampler=batch_sampler_train,collate_fn=collate_fn, num_workers=0)
val_dl = DataLoader(dataset_val, config.batch_size, sampler=sampler_val,drop_last=True, collate_fn=collate_fn, num_workers=0)


#print('dataload time:{:.0f}'.format((time.time()-start_time)/60))

model = network(config)

trainer = DetrTrainer(config, model, train_dl, val_dl,optimizer='Adam')
trainer.run()


# if __name__ == '__main__':
#     from torchsummary import summary
#     # 注释前面
#     model = network(config)
#     summary(model.cuda(),(1024,100),batch_size=2)
    # 查看model para size 和 合适的batch_size

#     for samples, targets in train_dl:
#         print('--------------')
#         print(samples[0].size())
#         print(len(targets))
#         print(targets[0])
#         print(targets[1])


