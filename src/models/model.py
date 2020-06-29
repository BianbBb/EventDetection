# framework
# assemble modules defined in other scripts

from .dbg.dbg import DBG
# from .detr.detr import DETR


def network(config):
    model = None
    if config.model_name == 'DBG':
        model = DBG(feature_dim=1024)
    elif config.model_name == 'DETR':
        model = None
    else:
        print('wrong model')
    return model


