import torch
import numpy as np
from torch import nn
from .modules import DSBaseNet, ProposalFeatureGeneration, ACRNet, TBCNet
from .layers import conv1d


class DBG(nn.Module):
    """
    Setup dense boundary generator framework (DBG)
    """

    def __init__(self, feature_dim):
        super(DBG, self).__init__()

        self.DSBNet = DSBaseNet(feature_dim)
        self.PropFeatGen = ProposalFeatureGeneration()
        self.ACRNet = ACRNet()
        self.TBCNet = TBCNet()

        self.best_loss = 999999
        self.reset_params()  # reset all params by glorot uniform

    @staticmethod
    def glorot_uniform_(tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        scale = 1.0
        scale /= max(1., (fan_in + fan_out) / 2.)
        limit = np.sqrt(3.0 * scale)
        return nn.init._no_grad_uniform_(tensor, -limit, limit)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
            DBG.glorot_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        DSB_output = self.DSBNet(x)
        action_feat, net_feat = self.PropFeatGen(DSB_output['score'], DSB_output['xc_feat'])
        iou = self.ACRNet(action_feat)
        prop_start, prop_end = self.TBCNet(net_feat)

        output_dict = {
            'x1': DSB_output['x1'],
            'x2': DSB_output['x2'],
            'x3': DSB_output['x3'],
            'iou': iou,
            'prop_start': prop_start,
            'prop_end': prop_end
        }
        return output_dict


class DBG_reduce_dim(nn.Module):
    def __init__(self, in_dim=1024, out_dim=400):
        super(DBG_reduce_dim, self).__init__()
        self.pre1 = conv1d(in_dim, out_dim // 2, 3)
        self.pre2 = conv1d(in_dim, out_dim // 2, 3)
        self.basemodel = DBG(feature_dim=400)

    def forward(self, x):
        p1 = self.pre1(x)  # (b,1024,100) ->(b,200,100)
        p2 = self.pre2(x)
        input = torch.cat([p1, p2], dim=1)  # (b,400,100)
        output = self.basemodel(input)
        return output
