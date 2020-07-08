import torch
from torch.utils.data import DataLoader

from models.model import network
from utils.read_config import Config
from workers.detr_dataloader import MyDataSet
from workers.detr_trainer import DetrTrainer

# prepare for training
if not torch.cuda.is_available():
    print('Only train on CPU.')

config = Config()


# dataset
# dataset_train = MyDataSet(config, mode='training')
# dataset_val = MyDataSet(config, mode='validation')

# if args.distributed:
#     sampler_train = DistributedSampler(dataset_train)
#     sampler_val = DistributedSampler(dataset_val, shuffle=False)
# else:
# sampler_train = torch.utils.data.RandomSampler(dataset_train)
# sampler_val = torch.utils.data.SequentialSampler(dataset_val)
#
# batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, config.batch_size, drop_last=True)
# batch_sampler_val = torch.utils.data.BatchSampler(sampler_train, config.batch_size, drop_last=False)

train_dl = DataLoader(MyDataSet(config, mode='training'), batch_size=config.batch_size,shuffle=True, num_workers=0, drop_last=True)
#val_dl = DataLoader(MyDataSet(config, mode='validation'), batch_size=config.batch_size,shuffle=False, num_workers=0, drop_last=True)


#
# # model
# model = network(config)
#
# trainer = DetrTrainer(config, model, train_dl, val_dl)
# trainer.run()


if __name__ == '__main__':
    for n_iter, (samples, targets) in enumerate(train_dl):
        print('--------------')
        print(len(targets))
        for k,v in targets.items():
            print(k)
            print(v)