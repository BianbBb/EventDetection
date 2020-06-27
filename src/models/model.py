# framework
# assemble modules defined in other scripts

import torch
import numpy as np
from src.models.dbg import DBG, DBG_reduce_dim


def network(config):
    model = None
    if config.model_name == 'DBG':
        model = DBG(feature_dim=400)
    elif config.model_name == 'DBG_reduce_dim':
        model = DBG_reduce_dim(in_dim=1024, out_dim=400)
    else:
        print('wrong model')
    return model


if __name__ == '__main__':
    net = network()
    x = torch.zeros([32, 1024, 100])
    print(net(x))
