import torch.nn as nn
from layers.transformer_layer import TransformerEncoderLayer
from layers.transformer_embeddings import TransformerEmbedding


class BERT(nn.Module):

    def __init__(self,
                 n_vocab,
                 n_blocks=12,
                 n_heads=8,
                 d_model=512,
                 d_ff=2048,
                 max_seq_len=256,
                 dropout=0.1,
                 padding_idx=None):
        super(BERT, self).__init__()
        self.n_vocab = n_vocab
        self.transformer_embeddings = TransformerEmbedding(n_vocab, d_model,
                                                           dropout=dropout,
                                                           padding_idx=padding_idx)

        self.transformer_blocks = nn.ModuleList()
        for i in range(n_blocks):
            block = TransformerEncoderLayer(
                d_model,
                n_heads,
                d_ff,
                dropout
            )
            self.transformer_blocks.append(block)

    def forward(self, inputs, segments=None):
        x = self.transformer_embeddings(inputs, segments)

        for block in self.transformer_blocks:
            x = block(x)

        return x
