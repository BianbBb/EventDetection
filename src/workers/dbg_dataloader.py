import torch
from torch.utils.data import Dataset
from utils.utils import getDatasetDict, getFullData
import numpy as np



class MyDataSet(Dataset):

    def __init__(self, config, mode='training'):

        video_info_file = config.video_info_file
        video_filter = config.video_filter
        data_aug = config.data_aug
        train_dict, val_dict, test_dict = getDatasetDict(config, video_info_file, video_filter)
        training = True
        if mode == 'training':
            video_dict = train_dict
            video_dict = dict(list(video_dict.items())[:100]) # TODO：comment out this line

        else:
            training = False
            video_dict = val_dict
            video_dict = dict(list(video_dict.items())[:100])  # TODO：comment out this line

        self.mode = mode
        self.video_dict = video_dict

        video_num = len(list(video_dict.keys()))

        video_list = np.arange(video_num)

        # load raw data
        if training:
            data_dict, train_video_mean_len = getFullData(config, video_dict, last_channel=False, training=True)
        else:
            data_dict = getFullData(config, video_dict, last_channel=False, training=False)

        # transform data to torch tensor
        for key in list(data_dict.keys()):
            data_dict[key] = torch.Tensor(data_dict[key]).float()
        self.data_dict = data_dict

        if data_aug and training:
            # add train video with short proposals
            add_list = np.where(np.array(train_video_mean_len) < 0.2)
            add_list = np.reshape(add_list, [-1])
            video_list = np.concatenate([video_list, add_list[:]], 0)

        self.video_list = video_list
        np.random.shuffle(self.video_list)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.video_list[idx]
        data_dict = self.data_dict
        gt_action = data_dict['gt_action'][idx].unsqueeze(0)
        gt_start = data_dict['gt_start'][idx].unsqueeze(0)
        gt_end = data_dict['gt_end'][idx].unsqueeze(0)
        feature = data_dict['feature'][idx]
        iou_label = []
        iou_label = data_dict['iou_label'][idx].unsqueeze(0)

        return gt_action, gt_start, gt_end, feature, iou_label

