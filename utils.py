import copy
import torch
import math
import torch.nn as nn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layer norm module "

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class SublayerConnection(nn.Module):
    """
    sublayerconnection is a building block of an encoder layer and a decoder layer.
    The encoder layer has two sublayers: self-attn, and feedforwrad.
    The decoder has three sublayers: self-attn, cross-attn, and feedforward.
    Each sublayer applies a layer normalization and a residual connection.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        Input sublayer can be self-attn or feedforward.
        """

        return x + self.dropout(sublayer(self.norm(x)))