import os

import numpy as np
import tqdm
import torch
import torch.nn as nn

from model import DBG
from utils.util import gen_mask, getBatchListTest, getProposalDataTest, save_proposals_result, getDatasetDict
from utils.parse_yaml import Config

torch.backends.cudnn.enabled = False
config = Config()
if not os.path.exists(config.results_dir):
    os.makedirs(config.results_dir)
pth_load_dir = config.test_pth_load_dir

save_dir = config.test_csv_save_dir
tscale = config.tscale
feature_dim = config.feature_dim
batch_size = config.test_batch_size

mask = gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), 1)
mask = torch.from_numpy(mask).float().requires_grad_(False).cuda()

"""
This test script is used for evaluating our algorithm 
This script saves all proposals results (csv format)
Then, use post_processing.py to generate the final result
Finally, use eval.py to evaluate the final result
You can got about 68% AUC
"""

""" 
Testing procedure
1.Get Test data
2.Define DBG model
3.Load model weights 
4.Run DBG model
5.Save proposal results (csv format)
"""


def test():
    with torch.no_grad():
        """ setup DBG model and load weights """
        net = DBG(feature_dim)
        state_dict = torch.load(os.path.join(pth_load_dir, 'checkpoint_best.pth'))
        net.load_state_dict(state_dict)
        net = nn.DataParallel(net, device_ids=[0]).cuda()
        net.eval()

        train_dict, val_dict, test_dict = getDatasetDict(config, config.video_info_file)

        if config.mode == 'validation':
            test_dict = val_dict
        batch_video_list = getBatchListTest(test_dict, batch_size)

        batch_result_xmin = []
        batch_result_xmax = []
        batch_result_iou = []
        batch_result_pstart = []
        batch_result_pend = []

        print('Runing DBG model ...')
        print("testing on {} dataset".format(config.mode))
        for idx in tqdm.tqdm(range(len(batch_video_list))):
            batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature = getProposalDataTest(config, batch_video_list[idx])
            in_feature = torch.from_numpy(batch_anchor_feature).float().cuda().permute(0, 2, 1)
            output_dict = net(in_feature)
            out_iou = output_dict['iou']
            out_start = output_dict['prop_start']
            out_end = output_dict['prop_end']

            # fusion starting and ending map score
            out_start = out_start * mask
            out_end = out_end * mask
            out_start = torch.sum(out_start, 3) / torch.sum(mask, 3)
            out_end = torch.sum(out_end, 2) / torch.sum(mask, 2)

            batch_result_xmin.append(batch_anchor_xmin)
            batch_result_xmax.append(batch_anchor_xmax)
            batch_result_iou.append(out_iou[:, 0].cpu().detach().numpy())
            batch_result_pstart.append(out_start[:, 0].cpu().detach().numpy())
            batch_result_pend.append(out_end[:, 0].cpu().detach().numpy())

        save_proposals_result(batch_video_list,
                              batch_result_xmin,
                              batch_result_xmax,
                              batch_result_iou,
                              batch_result_pstart,
                              batch_result_pend,
                              tscale,
                              save_dir)


if __name__ == "__main__":
    test()
