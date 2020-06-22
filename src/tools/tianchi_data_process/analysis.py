"""
三个数据集的数据分析
视频长度 Segment的长度 每个Segment占视频的比例 每个视频的segment数目
"""


import json
import matplotlib.pyplot as plt
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


def draw(info,dataset):
    print('max:{}'.format(max(info)))  # 448 755

    bins = np.arange(0, 1.01, 0.05)
    if dataset =='tianchi':
        color = 'b'
    elif dataset == 'activitynet':
        color = 'fuchsia'
    plt.hist(info, bins, color=color, alpha=0.5)

    plt.xlabel('value')
    plt.ylabel('count')
    plt.title('{} segment ratio'.format(dataset))

    # plt.xlim(0, 200)  # 设置x轴分布范围
    plt.show()


if __name__ == '__main__':
    dataset_name = 'activitynet'
    if dataset_name =='tianchi':
        info_file = '../../../data/Tianchi/train_annotations_new.json'
    elif dataset_name=='activitynet':
        info_file = '../../../data/ActivityNet/video_info_19993.json'
    elif dataset_name=='thumos':
        info_file = None

    info = load_info(info_file)
    lengths,act_labels,segment_num,segment_ratio = statistic(info)
    draw(segment_ratio,dataset_name)
