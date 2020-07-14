  # load net
    # load weight
# load feature
# tester.run()
# tester get output
# tester change the outputs to result

import os

import numpy as np
import tqdm
import torch
import torch.nn as nn

from models.model import network
from utils.utils import gen_mask, getBatchListTest, getProposalDataTest, save_proposals_result, getDatasetDict
from utils.read_config import Config

config = Config()
torch.backends.cudnn.enabled = False

class ADTR_tester():
    def __init__(self, config, net, test_loader):
        pass

    def load_weight(self):
        pass

    def forward(self):
        pass

    def post_process(self):
        pass

    def run(self):
        pass
