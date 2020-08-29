from evaluation.eval_proposal import ANETproposal
from evaluation.eval_detection import ANETdetection
from evaluation.eval_classification import ANETclassification
import numpy as np
import argparse

from utils.read_config import Config
config = Config()

""" Define parser """
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str,default='pro')  # det:detection  pro:proposal  class: classification
args = parser.parse_args()


def run_evaluation_proposal(ground_truth_filename, proposal_filename,
                            max_avg_nr_proposals=100,
                            tiou_thresholds=np.linspace(0.5, 0.95, 10),
                            subset='validation'):
    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,subset=subset, verbose=True, check_status=False)
    anet_proposal.evaluate()

    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    return average_nr_proposals, average_recall, recall


def run_evaluation_detection(ground_truth_filename, detection_filename,
                             tiou_thresholds=np.linspace(0.5, 0.95, 10),
                             subset='validation'):
    anet_detection = ANETdetection(ground_truth_filename, detection_filename,tiou_thresholds=tiou_thresholds,
                                   subset=subset, verbose=False, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()
    return mAP, average_mAP


def run_evaluation_classification(ground_truth_filename, detection_filename, subset='validation'):
    anet_classification = ANETclassification(ground_truth_filename, detection_filename,subset=subset,verbose=True, check_status=False)
    anet_classification.evaluate()
    ap = anet_classification.ap
    hit_at_k = anet_classification.hit_at_k
    avg_hit_at_k = anet_classification.avg_hit_at_k
    return ap, hit_at_k, avg_hit_at_k


def eval_proposal():
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = \
        run_evaluation_proposal(gt_file, result_file,max_avg_nr_proposals=100,
                                tiou_thresholds=np.linspace(0.5, 0.95, 10),subset='validation')

    print("AR@1 is \t", np.mean(uniform_recall_valid[:, 0]))
    print("AR@5 is \t", np.mean(uniform_recall_valid[:, 4]))
    print("AR@10 is \t", np.mean(uniform_recall_valid[:, 9]))
    print("AR@100 is \t", np.mean(uniform_recall_valid[:, -1]))


def eval_detection():
    map, avg_map = run_evaluation_detection(gt_file, result_file,tiou_thresholds=np.linspace(0.5, 0.95, 10),subset='validation')
    print("---------------------------")
    print("mAP@0.5     is {:3.8f}".format(map[0]))
    print("mAP@0.7     is {:3.8f}".format(map[4]))
    print("mAP@0.9     is {:3.8f}".format(map[8]))
    print("mAP@0.95    is {:3.8f}".format(map[9]))

    print("mAP Average is {:3.8f}".format(avg_map))
    print("---------------------------")

def eval_classification():
    ap, hit_at_k, avg_hit_at_k = run_evaluation_classification(gt_file, result_file,subset='validation')
    # print("---------------------------")
    # print(ap)
    # print(hit_at_k)
    # print(avg_hit_at_k)
    # print("---------------------------")

if __name__ == '__main__':
    result_file = config.post_json_save_path  # 结果文件：-r --result
    print("eval in ", result_file)
    dataset = config.dataset_name

    if dataset == 'activitynet':
        gt_file = "../data/ActivityNet/video_info_19993.json"
    elif dataset == 'tianchi':
        gt_file = "../data/Tianchi/train_annotations_new.json"
    elif dataset == 'thumos':
        gt_file = "../data/THUMOS/XXX"
    else:
        raise IOError('Dataset name is unavailable ！')

    if args.mode == 'pro':
        eval_proposal()
    elif args.mode == 'det':
        eval_detection()
    elif args.mode == 'class':
        eval_classification()