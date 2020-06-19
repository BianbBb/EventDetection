# 三个数据集的数据分析
# 视频长度 Segment的长度 每个Segment占视频的比例 每个视频的segment数目

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def statistic(info):
    info=info
    lengths = []
    segment_ratio = []
    segment_num = []
    act_labels = []
    test_duration = []
    for k,v in info.items():
        if v['subset'] != 'testing':
            duration = v['duration']
            lengths.append(duration)
            seg_num = 0

            for act in v['annotations']:
                start = act['segment'][0]
                end = act['segment'][1]
                segment_ratio.append((end-start)/duration)
                act_labels.append(act['label'])
                seg_num += 1
            segment_num.append(seg_num)
        else:
            test_duration.append(v['duration'])

    return lengths,act_labels,segment_num,segment_ratio


def load_info(filename):
    with open(filename,'r') as f:
        info  = json.load(f)
    return info


def draw(info):
    bins = np.linspace(0, max(info), 10)


if __name__ == '__main__':
    info_file = '../../../data/Tianchi/train_annotations_new.json'
    # info_file = '../../../data/ActivityNet/video_info_19993.json'
    info = load_info(info_file)
    lengths,act_labels,segment_num,segment_ratio = statistic(info)
    print(lengths)
    draw(lengths)
