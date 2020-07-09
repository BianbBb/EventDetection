# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

import sys
sys.path.append("../../")
from utils.proposal_ops import distance_iou, cl2xy


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_classes: float = 1, cost_segments: float = 1, cost_diou: float = 1):
        """Creates the matcher

        Params:
            cost_classes: This is the relative weight of the classification error in the matching cost
            cost_segments: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_diou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_classes = cost_classes
        self.cost_segments = cost_segments
        self.cost_diou = cost_diou
        assert cost_classes != 0 or cost_segments != 0 or cost_diou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes, 1] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["classes"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_classes = outputs["classes"].flatten(0, 1).softmax(-1).long()  # [batch_size * num_queries, num_classes]
        out_segments = outputs["segments"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_classes = torch.cat([v["classes"].long() for v in targets])
        tgt_segments = torch.cat([v["segments"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_classes = -out_classes[:, tgt_classes]

        # Compute the L1 cost between boxes
        cost_segments = torch.cdist(out_segments, tgt_segments, p=1)
        # Compute the diou cost betwen segments
        cost_diou = -distance_iou(cl2xy(out_segments), cl2xy(tgt_segments))

        # Final cost matrix
        C = self.cost_segments * cost_segments + self.cost_classes * cost_classes + self.cost_diou * cost_diou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["segments"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(config):
    return HungarianMatcher(cost_classes=config.set_cost_classes, cost_segments=config.set_cost_segments, cost_diou=config.set_cost_diou)
