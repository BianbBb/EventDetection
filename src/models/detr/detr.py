# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import build_transformer
from .position_encoding import PositionEmbedding


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, position_encoding, transformer, num_classes, num_queries, aux_loss=False):
        super(DETR, self).__init__()
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv1d(1024, hidden_dim, kernel_size=1)  # pre-process layer
        self.position_encoding = position_encoding  # position encoding layer, get encoding added to input
        self.transformer = transformer  # transformer layer
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object query
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # classifier for action
        self.proposal_embed = MLP(hidden_dim, hidden_dim, 2, 3)  # proposal regression
        self.aux_loss = aux_loss  # if use aux loss

    def forward(self, x):
        input_tensor = self.input_proj(x)
        pos_embed = self.position_encoding(input_tensor) - input_tensor
        hs = self.transformer(input_tensor, None, self.query_embed.weight, pos_embed)[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.proposal_embed(hs).sigmoid()
        out = {'classes': outputs_class[-1], 'segments': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'classes': a, 'segments': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_classes, out_segments = outputs['classes'], outputs['segments']

        assert len(out_classes) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_classes, -1)
        scores, classes = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        segments = out_segments
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        segments = segments * scale_fct[:, None, :]

        results = [{'scores': s, 'classes': l, 'segments': b} for s, l, b in zip(scores, classes, segments )]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_detr(config):
    position_encoding = PositionEmbedding(config.hidden_dim, 100)
    transformer = build_transformer(config)
    model = DETR(
        position_encoding,
        transformer,
        num_classes=53,
        num_queries=config.num_queries,
        aux_loss=False,
    )
    return model