"""
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
import parse_yaml


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


def split_dataset(raw_folder, train_folder, valid_folder, train_split=0.8):
    raw_train_folder = os.path.join(raw_folder, 'train', "i3d")
    f = open("../../data/train_annotations.json", 'r')
    content = f.readline()  # this file only one line
    annotation = json.loads(content)
    # total_size = len(annotation)
    total_size = 109
    print("total file", total_size)
    raw_index = np.arange(total_size)
    np.random.shuffle(raw_index)  # 随机打乱下标数组
    train_index = raw_index[:int(train_split * total_size)]  # 打乱的训练集
    valid_index = raw_index[int(train_split * total_size):]  # 打乱的测试集
    print("train videos num {}, valid videos num {}".format(len(train_index), len(valid_index)))
    train_id, train_annotation = [], []
    valid_id, valid_annotation = [], []
    keys = list(annotation.keys())
    for item in train_index:
        train_id.append(keys[item])
        shutil.copy(os.path.join(raw_train_folder, keys[item] + ".npy"),
                    os.path.join(train_folder, keys[item] + ".npy"))
        train_annotation.append(annotation[keys[item]])
    for item in valid_index:
        valid_id.append(keys[item])
        shutil.copy(os.path.join(raw_train_folder, keys[item] + ".npy"),
                    os.path.join(valid_folder, keys[item] + ".npy"))
        valid_annotation.append(annotation[keys[item]])
    pd.DataFrame({'file_id': train_id, 'annotation': train_annotation}).to_csv("../../data/train.csv", encoding="utf8",
                                                                               index=False)
    pd.DataFrame({'file_id': train_id, 'annotation': train_annotation}).to_csv("../../data/test.csv", encoding="utf8",
                                                                               index=False)


if __name__ == '__main__':
    config = parse_yaml.read_config("../../config/default.yml")  # config file
    root_folder = config['data']['root_folder']
    train_folder, valid_folder = os.path.join(root_folder, "train_split"), os.path.join(root_folder, "valid_split")
    check_folder(train_folder, valid_folder)
    split_dataset(root_folder, train_folder, valid_folder)
