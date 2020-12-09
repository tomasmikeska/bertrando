import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):

    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.query_layer = nn.Linear(d_model, d_model, bias=False)
        self.key_layer = nn.Linear(d_model, d_model, bias=False)
        self.value_layer = nn.Linear(d_model, d_model, bias=False)
        self.final_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def self_attention(self, query, key, value):
        '''Scaled Dot Product Attention'''
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value)

    def forward(self, x):
        batch_size = x.size(0)

        # Do all the linear projections in batch from d_model => h x d_k
        query = self.query_layer(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.key_layer(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.value_layer(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention on all the projected vectors in batch.
        x = self.self_attention(query, key, value)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.final_proj(x)
