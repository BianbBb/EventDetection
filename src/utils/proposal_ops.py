"""
Author: Zhou Chen
Date: 2020/7/1
Desc: desc
"""


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