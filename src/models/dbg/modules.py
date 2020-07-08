import torch
import torch.nn as nn
from ..custom_op.prop_tcfg_op import PropTcfg
from .layers import conv1d, conv2d


class DSBaseNet(nn.Module):
    """
    Setup dual stream base network (DSB)
    """

    def __init__(self, feature_dim):
        super(DSBaseNet, self).__init__()
        feature_dim = feature_dim // 2
        self.feature_dim = feature_dim
        self.conv1_1 = conv1d(feature_dim, 256, 3)
        self.conv1_2 = conv1d(256, 128, 3)
        self.conv1_3 = conv1d(128, 1, 1, is_relu=False)

        self.conv2_1 = conv1d(feature_dim, 256, 3)
        self.conv2_2 = conv1d(256, 128, 3)
        self.conv2_3 = conv1d(128, 1, 1, is_relu=False)

        self.conv3 = conv1d(128, 1, 1, is_relu=False)

    def forward(self, x):
        x1, x2 = torch.split(x, self.feature_dim, 1)
        x1 = self.conv1_1(x1)
        x1 = self.conv1_2(x1)
        x1_feat = x1
        x1 = torch.sigmoid(self.conv1_3(x1))

        x2 = self.conv2_1(x2)
        x2 = self.conv2_2(x2)
        x2_feat = x2
        x2 = torch.sigmoid(self.conv2_3(x2))

        xc = x1_feat + x2_feat
        xc_feat = xc
        x3 = torch.sigmoid(self.conv3(xc))

        score = (x1 + x2 + x3) / 3.0

        output_dict = {
            'score': score,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'xc_feat': xc_feat
        }
        return output_dict


class ProposalFeatureGeneration(nn.Module):
    """
    Setup proposal feature generation module
    """

    def __init__(self, in_channels=128):
        super(ProposalFeatureGeneration, self).__init__()
        self.prop_tcfg = PropTcfg()
        self.conv3d = nn.Conv3d(in_channels, 512, kernel_size=(32, 1, 1))

    def forward(self, action_score, xc_feat):
        action_feat = self.prop_tcfg(action_score)  # B x 1 x 32 x T x T
        action_feat = torch.squeeze(action_feat, 1)  # B x 32 x T x T
        net_feat = self.prop_tcfg(xc_feat)  # B x 128 x 32 x T x T
        net_feat = self.conv3d(net_feat)  # B x 512 x 1 x T x T
        net_feat = torch.squeeze(net_feat, 2)  # B x 512 x T x T

        return action_feat, net_feat


class ACRNet(nn.Module):
    """
    Setup action classification regression network (ACR)
    """

    def __init__(self, in_channels=32):
        super(ACRNet, self).__init__()
        self.conv2d = nn.Sequential(
            conv2d(in_channels, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 1, 1, is_relu=False)
        )

    def forward(self, action_feat):
        iou = self.conv2d(action_feat)
        iou = torch.sigmoid(iou)
        return iou


class TBCNet(nn.Module):
    """
    Setup temporal boundary classification network (TBC)
    """

    def __init__(self, in_channels=512):
        super(TBCNet, self).__init__()
        self.conv2d = nn.Sequential(
            conv2d(in_channels, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 256, 1),
            nn.Dropout(p=0.3),
            conv2d(256, 2, 1, is_relu=False)
        )

    def forward(self, net_feat):
        x = self.conv2d(net_feat)
        x = torch.sigmoid(x)

        prop_start = x[:, :1].contiguous()
        prop_end = x[:, 1:].contiguous()
        return prop_start, prop_end
