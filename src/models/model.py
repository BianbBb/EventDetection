# framework
# assemble modules defined in other scripts

# from .dbg.dbg import DBG
from .detr.detr import build


def network(config):
    model = None
    if config.model_name == 'DBG':
        model = None
    elif config.model_name == 'DETR':
        model = build(config)
    else:
        print('wrong model')
    print(model)
    return model


