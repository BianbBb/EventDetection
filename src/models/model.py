# framework
# assemble modules defined in other scripts

from .dbg import DBG, DBG_reduce_dim


def network(config):
    model = None
    if config.model_name == 'DBG':
        model = DBG(feature_dim=1024)
    elif config.model_name == 'DBG_reduce_dim':
        model = DBG_reduce_dim(in_dim=1024, out_dim=400)
    else:
        print('wrong model')
    return model


