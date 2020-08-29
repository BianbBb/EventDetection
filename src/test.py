import os

import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from models.model import network
from utils.utils import getBatchListTest, getProposalDataTest, save_proposals_result
from utils.read_config import Config
from workers.detr_dataloader import MyDataSet, getDatasetDict
from workers.detr_tester import ADTR_tester

torch.backends.cudnn.enabled = False
config = Config()


def save(result, file):
    (filepath, filename) = os.path.split(file)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    print('save to :{}'.format(file))
    with open(file, 'w') as f:
        json.dump(result, f)


def test(config):
    if not torch.cuda.is_available():
        print('Only test on CPU.')

    def collate_fn(batch):
        batch = list(zip(*batch))
        return tuple(batch)

    # dataset
    train_dict, val_dict, test_dict = getDatasetDict(config.video_info_file, config.video_filter)

    # test dataset无标注数据，只能使用val数据集进行evaluation
    if config.mode == 'validation':
        dataset_test = MyDataSet(config, val_dict, flag='test')
    elif config.mode == 'test':
        dataset_test = MyDataSet(config, test_dict, flag='test')
    else:
        print("the mode in default.yaml is wrong ! ")
    test_dl = DataLoader(dataset_test, config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=8)

    model = network(config)
    tester = ADTR_tester(config, model, test_dl)
    result = tester.run()
    output_file = config.post_json_save_path
    save(result, output_file)

if __name__ == '__main__':
    test(config)
