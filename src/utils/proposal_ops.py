"""
Author: Zhou Chen
Date: 2020/7/1
Desc: desc
"""

import torch

def proposal_iou(proposal1, proposal2):
    """
    iou with two proposal
    :param proposal1:
    :param proposal2:
    :return:
    """
    s1, e1 = proposal1
    s2, e2 = proposal2
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor

def distance_iou(seg1, seg2): # 值域为[-1,1]
    # TODO:删除 losses 中定义的func
    assert (seg1[..., 0] >= seg1[...,1]).all()
    assert (seg2[..., 0] >= seg2[...,1]).all()
    inter = torch.max(seg1[...,1], seg2[...,1]) - torch.min(seg1[...,0], seg2[...,0])# 交集
    union = (torch.min(seg1[...,1], seg2[...,1]) - torch.max(seg1[...,0], seg2[...,0])).clamp(min=0) # 并集
    center_distance = torch.abs(seg1[...,1]+seg1[...,0]-seg2[...,1]-seg2[...,0])/2 # 中心距离
    diou = (inter - center_distance)/union
    return diou