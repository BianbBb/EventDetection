"""
To spilt the training dataset to train & val set,
and update the annotation file by adding subset and duration information.
Author: Zhou Chen
Date: 2020/6/9
Desc: split data
"""
import os
import shutil
import sys
import json

import numpy as np
import pandas as pd

np.random.seed(2020)

sys.path.append("../utils/")

def check_folder(train_folder, valid_folder):
    """
    创建训练集和测试集根目录
    :param train_folder:
    :param valid_folder:
    :return:
    """
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(valid_folder):
        os.mkdir(valid_folder)


def get_duration(video_id, folder):
    return( len(np.load(os.path.join(folder,"{}.npy".format(video_id) ))) *8 /15 )


def split_dataset(raw_folder, old_path, new_path, npy_folder, train_split=0.8):
    f_old = open(old_path, 'r')
    content = f_old.readline()  # this file only one line
    annotations = json.loads(content)
    total_size = len(annotations)
    print("total file", total_size)

    raw_index = np.arange(total_size)
    np.random.shuffle(raw_index)  # 随机打乱下标数组
    train_index = raw_index[:int(train_split * total_size)]  # shuffled train dataset
    valid_index = raw_index[int(train_split * total_size):]  # shuffled validation dataset
    print("train videos num {}, valid videos num {}".format(len(train_index), len(valid_index)))

    new_annotaions = annotations
    keys = list(annotations.keys())
    f_old.close()

    f_new = open(new_path, 'w')

    for item in train_index:
        item_id = keys[item]
        new_annotaions[item_id]['subset'] = 'training'
        new_annotaions[item_id]['duration'] = get_duration(item_id, npy_folder)

    for item in valid_index:
        item_id = keys[item]
        new_annotaions[item_id]['subset'] = 'validation'
        new_annotaions[item_id]['duration'] = get_duration(item_id, npy_folder)
    json.dump(new_annotaions,f_new)
    f_new.close()

if __name__ == '__main__':
    data_folder = "/data/byh//EventDetection/train/i3d/"
    info_folder = "../../data/Tianchi/"
    old_anno_path = os.path.join(info_folder, "train_annotations.json")
    new_anno_path = os.path.join(info_folder, "train_annotations_new.json")
    split_dataset(info_folder, old_anno_path, new_anno_path,data_folder)
