import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))


class ChunkedFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1, n_chunks=16):
        super(ChunkedFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.n_chunks = n_chunks

    def forward_chunk(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))

    def forward(self, x):
        chunked_x = x.chunk(self.n_chunks, dim=1)
        chunked_y = [self.forward_chunk(x) for x in chunked_x]
        return torch.cat(chunked_y, dim=1)


class GLU(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.v_1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x, **kwargs):
        return self.w_2(self.act(self.w_1(x)) * self.v_1(x))
