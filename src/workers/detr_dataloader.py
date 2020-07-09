import torch
from torch.utils.data import Dataset
from utils.utils import get_filter_video_names, load_json,load_feature
import numpy as np
import json
import sys
sys.path.append("../")
from utils.proposal_ops import xy2cl


def getDatasetDict(config, video_info_file, video_filter=False):
    json_data = load_json(video_info_file)
    filter_video_names = get_filter_video_names(video_info_file)  # load filter video name

    database = json_data
    train_dict = {}
    val_dict = {}
    test_dict = {}
    video_lists = list(json_data.keys())
    for video_name in video_lists[:]:
        if video_filter and video_name in filter_video_names:
            continue
        video_info = database[video_name]
        video_new_info = dict()
        video_new_info["duration_second"] = video_info["duration"]
        video_subset = video_info['subset']
        video_new_info["annotations"] = video_info["annotations"]
        if video_subset == "training":
            train_dict[video_name] = video_new_info
        elif video_subset == "validation":
            val_dict[video_name] = video_new_info
        elif video_subset == "testing":
            test_dict[video_name] = video_new_info
    return train_dict, val_dict, test_dict


def getFullData(config, video_dict ,classes_index,last_channel=False, training=True):
    tscale = config.tscale
    data_dir = config.feat_dir
    video_list = list(video_dict.keys())

    batch_anchor_feature = []
    batch_label_action = []
    batch_label_start = []
    batch_label_end = []

    train_video_mean_len = []

    for i in range(len(video_list)): #TODO:tqdm 每一个video
        if i % 100 == 0:
            print("%d / %d videos are loaded" % (i, len(video_list)))
        video_name = video_list[i]
        video_info = video_dict[video_name]
        video_second = video_info["duration_second"]
        video_infos = video_info["annotations"]

        video_labels = []
        video_starts = []
        video_ends = []
        gt_lens = []
        for j in range(len(video_infos)): # video中的所有segment信息
            tmp_info = video_infos[j]
            tmp_label = tmp_info["label"]
            tmp_start = tmp_info["segment"][0]
            tmp_end = tmp_info["segment"][1]
            tmp_start = max(min(1, tmp_start / video_second), 0)
            tmp_end = max(min(1, tmp_end / video_second), 0)
            gt_lens.append(tmp_end - tmp_start)
            video_labels.append(classes_index[tmp_label])
            video_starts.append(tmp_start)
            video_ends.append(tmp_end)

        # calculate gt average length
        mean_len = 2
        if len(gt_lens):
            mean_len = np.mean(gt_lens)
        if training:
            train_video_mean_len.append(mean_len)

        # load feature
        video_feat = load_feature(config, data_dir, video_name)

        if not last_channel:
            video_feat = np.transpose(video_feat, [1, 0])
        batch_anchor_feature.append(video_feat)
        batch_label_action.append(video_labels)
        batch_label_start.append(video_starts)
        batch_label_end.append(video_ends)



    dataDict = {
        "gt_action": batch_label_action,
        "gt_start": batch_label_start,
        "gt_end": batch_label_end,
        "feature": batch_anchor_feature,
    }
    if training:
        return dataDict, train_video_mean_len
    else:
        return dataDict


class MyDataSet(Dataset):
    def __init__(self, config, mode='training'):
        video_info_file = config.video_info_file
        video_filter = config.video_filter
        data_aug = config.data_aug
        train_dict, val_dict, test_dict = getDatasetDict(config, video_info_file, video_filter)
        training = True
        if mode == 'training':
            video_dict = train_dict
            # video_dict = dict(list(video_dict.items())[:500]) # TODO：comment out this line

        else:
            training = False
            video_dict = val_dict
            # video_dict = dict(list(video_dict.items())[:500])  # TODO：comment out this line

        self.mode = mode
        self.video_dict = video_dict

        with open(config.index_file,'r') as f:
            self.classes_index = json.load(f)

        video_num = len(list(video_dict.keys()))

        video_list = np.arange(video_num)

        # load raw data
        if training: ##############
            data_dict, train_video_mean_len = getFullData(config, video_dict, self.classes_index,last_channel=False, training=True)
        else:##############
            data_dict = getFullData(config, video_dict, self.classes_index, last_channel=False, training=False)

        # transform data to torch tensor
        # for key in list(data_dict.keys()):
        #     data_dict[key] = torch.Tensor(data_dict[key]).float()
        self.data_dict = data_dict

        # if data_aug and training:
        #         #     # add train video with short proposals
        #         #     add_list = np.where(np.array(train_video_mean_len) < 0.2)
        #         #     add_list = np.reshape(add_list, [-1])
        #         #     video_list = np.concatenate([video_list, add_list[:]], 0)

        self.video_list = video_list
        np.random.shuffle(self.video_list)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.video_list[idx]
        data_dict = self.data_dict
        gt_action = data_dict['gt_action'][idx]
        gt_start = data_dict['gt_start'][idx]
        gt_end = data_dict['gt_end'][idx]
        feature = torch.Tensor(data_dict['feature'][idx])

        tmp_segment = []
        for i, j in zip(gt_start, gt_end):
            tmp_segment.append( [i, j])

        gt_segment = xy2cl(torch.Tensor(tmp_segment)).numpy().tolist()

        target = {'classes':gt_action, 'segments':gt_segment}
        return feature,target

