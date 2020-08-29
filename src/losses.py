import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.misc import accuracy, is_dist_avail_and_initialized, get_world_size
from utils.proposal_ops import distance_iou,cl2xy


def binary_logistic_loss(gt_scores, pred_anchors):
    """
    Calculate weighted binary logistic loss
    :param gt_scores: gt scores tensor
    :param pred_anchors: prediction score tensor
    :return: loss output tensor
    """
    gt_scores = gt_scores.view(-1)
    pred_anchors = pred_anchors.view(-1)

    pmask = (gt_scores > 0.5).float()
    num_positive = torch.sum(pmask)
    num_entries = pmask.size()[0]

    ratio = num_entries / max(num_positive, 1)
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 1e-6
    neg_pred_anchors = 1.0 - pred_anchors + epsilon
    pred_anchors = pred_anchors + epsilon

    loss = coef_1 * pmask * torch.log(pred_anchors) + coef_0 * (1.0 - pmask) * torch.log(
        neg_pred_anchors)
    loss = -1.0 * torch.mean(loss)
    return loss

'''
def IoU_loss(gt_iou, pred_iou, mask):  #todo :
    """
    Calculate IoU loss
    :param gt_iou: gt IoU tensor
    :param pred_iou: prediction IoU tensor
    :return: loss output tensor
    """
    u_hmask = (gt_iou > 0.6).float()
    u_mmask = ((gt_iou <= 0.6) & (gt_iou > 0.2)).float()
    u_lmask = (gt_iou <= 0.2).float() * mask

    u_hmask = u_hmask.view(-1)
    u_mmask = u_mmask.view(-1)
    u_lmask = u_lmask.view(-1)

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = 1.0 * num_h / num_m
    r_m = torch.min(r_m, torch.Tensor([1.0]).cuda())
    u_smmask = torch.rand(u_hmask.size()[0], requires_grad=False).cuda() * u_mmask
    u_smmask = (u_smmask > (1.0 - r_m)).float()

    r_l = 2.0 * num_h / num_l
    r_l = torch.min(r_l, torch.Tensor([1.0]).cuda())

    u_slmask = torch.rand(u_hmask.size()[0], requires_grad=False).cuda() * u_lmask
    u_slmask = (u_slmask > (1.0 - r_l)).float()

    iou_weights = u_hmask + u_smmask + u_slmask

    gt_iou = gt_iou.view(-1)
    pred_iou = pred_iou.view(-1)

    iou_loss = F.smooth_l1_loss(pred_iou * iou_weights, gt_iou * iou_weights, reduction='none')
    iou_loss = torch.sum(iou_loss * iou_weights) / torch.max(torch.sum(iou_weights),
                                                             torch.Tensor([1.0]).cuda())
    return iou_loss
'''


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        B, Q, N = inputs.size()  # batch query class_num+1
        P = F.softmax(inputs, dim=-1)

        class_mask = inputs.data.new(B, Q, N).fill_(0)
        ids = targets.unsqueeze(-1)
        class_mask.scatter_(-1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[ids.data.squeeze(-1)]
        probs = (P * class_mask).sum(-1).unsqueeze(-1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)


    def loss_classes(self, outputs, targets, indices, num_segments):
        """Classification loss (NLL)
        targets dicts must contain the key "classes" containing a tensor of dim [nb_target_segments]
        """
        assert 'classes' in outputs

        # alpha = torch.ones(self.num_classes+1, 1)
        # alpha[self.num_classes] = 1e-5
        focal_loss = FocalLoss(class_num=self.num_classes+1, size_average=True)

        #pred_classes = outputs['classes'].softmax(-1)
        pred_classes = outputs['classes']
        idx = self._get_src_permutation_idx(indices) # batch_idx, src_idx

        #batch_idx = idx[0].tolist()
        target_classes_o = torch.cat([t["classes"][J] for t, (_, J) in zip(targets, indices)]) #获得indices对应的segment的实际动作类别
        target_classes_o = target_classes_o.to(dtype=torch.int64)
        target_classes = torch.full(pred_classes.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=pred_classes.device)
        target_classes[idx] = target_classes_o

        loss_class = F.cross_entropy(pred_classes.transpose(1, 2), target_classes, self.empty_weight)
        # loss_class = focal_loss(pred_classes, target_classes)
        losses = {'loss_class': loss_class}
        losses['class_error'] = 100 - accuracy(pred_classes[idx], target_classes_o)[0]
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_segments):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty segments
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['classes']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["classes"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses



    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the bounding segments, the L1 regression loss and the DIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [100, 2]
           The target segments are expected in format (start, end), normalized by the feature length.
        """
        assert 'segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        pred_segments = outputs['segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # loss_segments = F.l1_loss(pred_segments, target_segments, reduction='none')
        loss_segments = F.smooth_l1_loss(pred_segments, target_segments, reduction='none')
        losses = {}
        losses['loss_segments'] = loss_segments.mean()

        loss_diou = 1 - torch.diag(distance_iou(cl2xy(pred_segments),cl2xy(target_segments)))
        # loss_diou = ((1 - distance_iou(pred_segments,target_segments))/2).sum()
        # losses['loss_diou'] = loss_diou.sum() / num_segments /2
        losses['loss_diou'] = loss_diou.mean()
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'classes': self.loss_classes,
            'segments': self.loss_segments,
            'cardinality': self.loss_cardinality,
            # 'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["classes"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized(): # distribute 分布式模型时使用
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'classes':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
