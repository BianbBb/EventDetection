from evaluation.eval_proposal import ANETproposal
from evaluation.eval_detection import ANETdetection
import numpy as np
import argparse

""" Define parser """
parser = argparse.ArgumentParser()
parser.add_argument("--gtfile", type=str)
parser.add_argument('--evalfile', type=str)
args = parser.parse_args()


def run_evaluation_proposal(ground_truth_filename, proposal_filename,
                   max_avg_nr_proposals=100,
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):
    """
    Proposal results evaluation
    :param ground_truth_filename: the path of ground truth json file
    :param proposal_filename: the path of proposal results json file
    :param max_avg_nr_proposals: maximum average number of proposal
    :param tiou_thresholds: list of tIoU thresholds
    :param subset: subset type
    :return: evaluation results
    """
    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=False)
    anet_proposal.evaluate()

    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video

    return average_nr_proposals, average_recall, recall


def run_evaluation_detection(ground_truth_filename, detection_filename,
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):
    anet_detection = ANETdetection(ground_truth_filename, detection_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 subset=subset, verbose=True, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()

    return mAP, average_mAP

gt_file = args.gtfile
eval_file = args.evalfile

uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = \
    run_evaluation_proposal(
        gt_file,
        eval_file,
        max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset='validation')

print("AR@1 is \t", np.mean(uniform_recall_valid[:, 0]))
print("AR@5 is \t", np.mean(uniform_recall_valid[:, 4]))
print("AR@10 is \t", np.mean(uniform_recall_valid[:, 9]))
print("AR@100 is \t", np.mean(uniform_recall_valid[:, -1]))

map, avg_map = run_evaluation_detection(gt_file, eval_file, tiou_thresholds=np.linspace(0.5, 0.95, 10), subset='validation')
