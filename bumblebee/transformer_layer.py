import torch.nn as nn
from ffn import PositionwiseFeedForward
from multihead_attention import MultiHeadedAttention


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadedAttention(n_heads, d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.self_attn(x))
        return self.norm2(x + self.ffn(x))
