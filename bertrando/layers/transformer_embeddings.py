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


class TransformerEmbedding(nn.Module):

    def __init__(self, n_vocab, embedding_size, dropout=0.2, n_segment_vocab=2, padding_idx=None):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(n_vocab, embedding_size, padding_idx=padding_idx)
        self.positional_embedding = LearnedPositionalEmbedding(embedding_size)
        self.segment_embedding = nn.Embedding(n_segment_vocab, embedding_size)
        self.embedding_norm = nn.LayerNorm(embedding_size)
        self.embedding_dropout = nn.Dropout(p=dropout)

    def forward(self, x, segments=None):
        x = self.token_embedding(x)
        x += self.positional_embedding(x)
        if segments is not None:
            x += self.segment_embedding(segments)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)
        return x
