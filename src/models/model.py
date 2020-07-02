# framework
# assemble modules defined in other scripts
from .dbg.dbg import DBG
from .detr.detr import build_detr


def network(config):
    model = None
    if config.model_name == 'DBG':
        model = DBG(config.feature_dim)
    elif config.model_name == 'DETR':
        model = build_detr(config)
    else:
        print('wrong model')
    return model


