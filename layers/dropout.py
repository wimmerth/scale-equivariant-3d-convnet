"""
Dropout layer for 6D tensors (B, C, S, X, Y, Z). Randomly dropouts entire channels across all scales.
"""
from torch import nn
import torch.nn.functional as F


class SEDropout(nn.Module):
    """Behaves like Dropout3D but eliminates same channels across all scales."""

    def __init__(self, p=0.5, inplace=False):
        super(SEDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        b, c, s, *xyz = x.shape
        x = x.view(b, c, s, -1)
        x = F.dropout2d(x, self.p, self.training, self.inplace)
        x = x.view(b, c, s, *xyz)
        return x
