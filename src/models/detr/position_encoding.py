"""
refer to https://github.com/CyberZHG/torch-position-embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def positionalencoding1d(d_model, length):
    """

    Args:
        d_model: dimension of the model
        length: length of positions
        batch_size:

    Returns:length*d_model position matrix

    """

    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros((length, d_model))
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = pe.permute(1,0)
    pe.requires_grad = False
    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len=5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        # x = x + weight[:x.size(0),:]
        # return self.dropout(x)
        return weight[:x.size(0),:]


if __name__ == '__main__':
    import cv2
    import numpy as np
    # emd = PositionEmbedding(100, 256)
    # x = torch.zeros((2, 100, 256))
    # out = emd(x)
    # print(out.size())
    #
    # zzz = emd(x)[0].detach().numpy()

    pe = positionalencoding1d(400,100)
    pe = pe.unsqueeze(0).repeat(32, 1, 1)
    # print(pe.size())
    pe = pe[0].cpu().numpy()

    zz = np.uint8((pe - np.min(pe)) * 255 / (np.max(pe) - np.min(pe)))
    # print(np.min(zz), np.max(zz))

    cv2.imshow('position encoding', zz)
    cv2.waitKey()