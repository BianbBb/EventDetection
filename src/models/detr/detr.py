# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import build_transformer
from .position_encoding import positionalencoding1d,PositionEmbedding


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, position_encoding, transformer, num_classes, num_queries, aux_loss=False, input_c=400):
        super(DETR, self).__init__()
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv1d(input_c, hidden_dim, kernel_size=1)  # pre-process layer
        self.position_encoding = position_encoding
        self.transformer = transformer  # transformer layer
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object query
        self.class_ffn = nn.Sequential(
            nn.Linear(hidden_dim, num_classes + 1),
            nn.LeakyReLU(),
            )
        # classifier for action
        #self.class_embed.bias.data.fill_(-2.19) # 使用focal loss时使用 根据数据集正负样本数目确定，coco数据集为0.0064:1 设置为-log((1-n)/n)
        self.proposal_ffn = MLP(hidden_dim, hidden_dim, 2, 3)  # proposal regression
        self.aux_loss = aux_loss  # if use aux loss

    def forward(self, x): # x:b,c,t 2,400,100
        # x = x.transpose(1, 2)
        b, c, t = x.size()
        input_tensor = self.input_proj(x) # 2,256,100


        pos_embed = self.position_encoding.unsqueeze(0).repeat(b, 1, 1) # 2,256,100
        pos_embed = pos_embed.cuda()
        # pos_embed = None # torch.zeros_like(input_tensor)

        hs = self.transformer(input_tensor, None, self.query_embed.weight, pos_embed)[0]
        # hs:6,2,100,256   decoder_layer_num, batch, query_num, feature_dim
        outputs_class = self.class_ffn(hs[-1])  #hs[-1] : 2,100,201
        outputs_segment = self.proposal_ffn(hs[-1]).sigmoid()
        out = {'classes': outputs_class, 'segments': outputs_segment}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segment)
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
    position_encoding = positionalencoding1d(config.hidden_dim, config.tscale)
    # position_encoding = PositionEmbedding(config.tscale,config.hidden_dim)
    transformer = build_transformer(config)
    model = DETR(
        position_encoding,
        transformer,
        num_classes=config.num_classes,
        num_queries=config.num_queries,
        aux_loss=False,
        input_c=config.feature_dim
    )
    return model