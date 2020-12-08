import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=2048):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return Variable(self.pe[:, :x.size(1)], requires_grad=False)


class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(LearnedPositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len).expand((1, -1)))

    def forward(self, x):
        position_ids = self.position_ids[:, :x.size()[1]]
        return self.embedding(position_ids)
