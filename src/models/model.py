# framework
# assemble modules defined in other scripts

from .dbg.dbg import DBG, DBG_reduce_dim
from .detr.detr import DETR


def network(config):
    model = None
    if config.model_name == 'DBG':
        model = DBG(feature_dim=1024)
    elif config.model_name == 'DBG_reduce_dim':
        model = DBG_reduce_dim(in_dim=1024, out_dim=400)
    elif config.model_name == 'DETR':
        model = DETR
    else:
        print('wrong model')
    return model


