"""
三个数据集的数据分析
视频长度 Segment的长度 每个Segment占视频的比例 每个视频的segment数目
"""
import json
import matplotlib.pyplot as plt
import numpy as np


def statistic(gt, result, subset="validation", contrast=False):
    # if contrast is True, show analysis about prediction result

    lengths = []
    segment_ratio = []
    segment_num = []
    act_labels = []
    test_duration = []
    for k, v in gt.items():
        if v['subset'] != subset:
            continue
        else:
            if v['subset'] != "testing":
                duration = v['duration']
                lengths.append(duration)
                seg_num = 0
                anno = result[k] if contrast else v["annotations"]
                for act in anno:
                    start = act['segment'][0]
                    end = act['segment'][1]
                    segment_ratio.append((end - start) / duration)
                    act_labels.append(act['label'])
                    seg_num += 1
                segment_num.append(seg_num)
            else:
                test_duration.append(v['duration'])

    return lengths, act_labels, segment_num, segment_ratio


def load_info(filename):
    with open(filename, 'r') as f:
        info = json.load(f)
    return info


def draw(info, dataset):
    print('max:{}'.format(max(info)))  # 448 755

    bins = np.arange(0, 1.01, 0.05)
    color = None
    if dataset == 'tianchi':
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
    dataset_name = 'tianchi'
    gt_file, result_file = None, None
    if dataset_name == 'tianchi':
        gt_file = '../../data/Tianchi/train_annotations_new.json'
        result_file = '../../results/DBG-0625-1115-tianchi-validation.json'
    elif dataset_name == 'activitynet':
        gt_file = '../../data/ActivityNet/video_info_19993.json'
        result_file = '../../../results/DBG-0623-1614-activitynet-validation.json'
    elif dataset_name == 'thumos':
        gt_file = None

    gt = load_info(gt_file)
    result = load_info(result_file)
    lengths, act_labels, segment_num, segment_ratio = statistic(gt, result, subset='validation', contrast=False)
    draw(segment_ratio, dataset_name)
