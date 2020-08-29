import torch
from torch.utils.data import Dataset
from utils.utils import get_filter_video_names, load_json, load_feature
import numpy as np
import json
import sys
from tqdm import tqdm

sys.path.append("../")
from utils.proposal_ops import xy2cl


def getDatasetDict(video_info_file, video_filter=False):
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


def getFullData(config, video_dict, classes_index,flag='train',last_channel=False, ):
    if flag == 'test':
        data_dir = config.test_dir
    else:
        data_dir = config.feat_dir

    video_list = list(video_dict.keys())

    batch_anchor_feature = []
    batch_label_action = []
    batch_label_start = []
    batch_label_end = []
    batch_video_name = []
    batch_video_duration = []

    for i in tqdm(range(len(video_list))):
        video_name = video_list[i]
        video_info = video_dict[video_name]
        video_second = video_info["duration_second"]
        video_infos = video_info["annotations"]
        batch_video_name.append(video_name)
        batch_video_duration.append(video_second)

        video_labels = []
        video_starts = []
        video_ends = []
        gt_lens = []

        if flag != 'test':
            for j in range(len(video_infos)):  # video中的所有segment信息
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

        batch_label_action.append(video_labels)
        batch_label_start.append(video_starts)
        batch_label_end.append(video_ends)

        video_feat = load_feature(config, data_dir, video_name)

        if not last_channel:
            video_feat = np.transpose(video_feat, [1, 0])

        batch_anchor_feature.append(video_feat)

    dataDict = {
        "gt_action": batch_label_action,
        "gt_start": batch_label_start,
        "gt_end": batch_label_end,
        "feature": batch_anchor_feature,
        "video_name": batch_video_name,
        "video_duration": batch_video_duration
    }

    return dataDict


class MyDataSet(Dataset):
    def __init__(self, config, video_dict, flag = 'train'):
        self.config = config
        self.flag = flag
        with open(config.index_file, 'r') as f:
            self.classes_index = json.load(f)

        # TODO：comment out this line
        # video_dict = dict(list(video_dict.items())[:51])

        self.video_num = len(video_dict.keys())
        data_dict = getFullData(config, video_dict, self.classes_index,flag=flag)
        self.data_dict = data_dict

    def __len__(self):
        return self.video_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_dict = self.data_dict

        feature = torch.from_numpy(data_dict['feature'][idx]).type(torch.FloatTensor)

        if self.flag != 'test':
            gt_action = data_dict['gt_action'][idx]
            gt_start = data_dict['gt_start'][idx]
            gt_end = data_dict['gt_end'][idx]
            tmp_segment = list(([i,j] for i, j in zip(gt_start, gt_end)))
            # for i, j in zip(gt_start, gt_end):
            #     tmp_segment.append([i, j])
            gt_segment = xy2cl(torch.Tensor(tmp_segment)).numpy().tolist() #TODO:xy2cl直接对列表操作
            target = {'classes': gt_action, 'segments': gt_segment}
        else:
            target = {}

        video_name = data_dict['video_name'][idx]
        video_duration = data_dict['video_duration'][idx]
        infos = {'video_name':video_name, 'video_duration':video_duration }

        return feature, target, infos
