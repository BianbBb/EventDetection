"""
refer to https://github.com/CyberZHG/torch-position-embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PositionEmbedding']


class PositionEmbedding(nn.Module):
    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self, num_embeddings, embedding_dim, mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )


if __name__ == '__main__':
    import cv2
    import numpy as np
    emd = PositionEmbedding(100, 1024)
    x = torch.zeros((32, 100, 1024))
    print(emd(x)[0], emd(x)[1])
    zzz = emd(x)[0].detach().numpy()
    zz = np.uint8((zzz-np.min(zzz))*255/(np.max(zzz)-np.min(zzz)))
    print(np.min(zz),np.max(zz))
    cv2.imshow('zz',zz)
    cv2.waitKey()
    print(emd(x)[1].size())
