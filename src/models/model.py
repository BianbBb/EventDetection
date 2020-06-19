# framework
# assemble modules defined in other scripts

import torch
from utils.parse_yaml import Config
from torch import nn


config = Config()

class network(nn.Module):
    def __init__(self):
        if config.model == 'DBG':
            network = DBG()
            return network
    def forword(self):
        pass


def DBG():
    return DBG_network()
