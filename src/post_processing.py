import argparse
import json
import multiprocessing as mp
import os
import threading

import numpy as np
import pandas as pd
import tqdm

from utils.util import getDatasetDict
from utils.parse_yaml import Config

config = Config()


""" Define parser """
parser = argparse.ArgumentParser()
parser.add_argument('top_number', type=int, nargs='?', default=100)
parser.add_argument('-t', '--thread', type=int, nargs='?', default=8)
args = parser.parse_args()

""" Number of proposal needed to keep for every video"""
top_number = args.top_number
""" Number of thread for post processing"""
thread_num = args.thread


def IOU(s1, e1, s2, e2):
    """
    Calculate IoU of two proposals
    :param s1: starting point of A proposal
    :param e1: ending point of A proposal
    :param s2: starting point of B proposal
    :param e2: ending point of B proposal
    :return: IoU value
    """
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def softNMS(df):
    """
    soft-NMS for all proposals
    :param df: input dataframe
    :return: dataframe after soft-NMS
    """
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []
    while len(tscore) > 1 and len(rscore) < top_number:
        max_index = tscore.index(max(tscore))
        tmp_start = tstart[max_index]
        tmp_end = tend[max_index]
        tmp_score = tscore[max_index]
        rstart.append(tmp_start)
        rend.append(tmp_end)
        rscore.append(tmp_score)
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

        tstart = np.array(tstart)
        tend = np.array(tend)
        tscore = np.array(tscore)

        tt1 = np.maximum(tmp_start, tstart)
        tt2 = np.minimum(tmp_end, tend)
        intersection = tt2 - tt1
        duration = tend - tstart
        tmp_width = tmp_end - tmp_start
        iou = intersection / (tmp_width + duration - intersection).astype(np.float)

        idxs = np.where(iou > 0.65 + 0.25 * tmp_width)[0]
        tscore[idxs] = tscore[idxs] * np.exp(-np.square(iou[idxs]) / 0.75)

        tstart = list(tstart)
        tend = list(tend)
        tscore = list(tscore)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def sub_processor(lock, pid, video_list):
    """
    Define job for every subprocess
    :param lock: threading lock
    :param pid: sub processor id
    :param video_list: video list assigned to each subprocess
    :return: None
    """
    text = 'processor %d' % pid
    with lock:
        progress = tqdm.tqdm(
            total=len(video_list),
            position=pid,
            desc=text
        )
    for i in range(len(video_list)):
        video_name = video_list[i]
        """ Read result csv file """
        df = pd.read_csv(os.path.join(config.result_dir, video_name + ".csv"))
        """ Calculate final score of proposals """
        df['score'] = df.iou.values[:] * df.start.values[:] * df.end.values[:]
        if len(df) > 1:
            df = softNMS(df)
        df = df.sort_values(by="score", ascending=False)
        video_info = video_dict[video_name]
        video_duration = video_info["duration_second"]
        proposal_list = []

        for j in range(min(top_number, len(df))):
            tmp_proposal = {}
            tmp_proposal["score"] = df.score.values[j]
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                       min(1, df.xmax.values[j]) * video_duration]
            #tmp_proposal["label"] = "驾驶汽车"  # TODO:label
            tmp_proposal["label"] = "Fun sliding down"
            proposal_list.append(tmp_proposal)
        result_dict[video_name[2:]] = proposal_list
        with lock:
            progress.update(1)

    with lock:
        progress.close()


if __name__ == '__main__':
    print("starting post processing")

    train_dict, val_dict, test_dict = getDatasetDict(config, config.video_info_file)
    print(len(test_dict.keys()))
    if config.test_mode == 'validation':
        video_dict = val_dict
    else:
        video_dict = test_dict

    results_dir = config.results_dir
    output_file = os.path.join(config.results_dir, "{}.json".format(config.test_pth_name))
    video_list = list(video_dict.keys())

    """ Post processing using multiprocessing
    """
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    total_video_num = len(video_list)
    per_thread_video_num = total_video_num // thread_num
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, sub_video_list))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    """ Save result json file """
    result_dict = dict(result_dict)

    with open(output_file, 'w') as outfile:
        json.dump(result_dict, outfile)
