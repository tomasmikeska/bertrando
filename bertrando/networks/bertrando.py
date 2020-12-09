import torch.nn as nn
from layers.transformer_layer import TransformerEncoderLayer
from layers.transformer_embeddings import TransformerEmbedding


class Bertrando(nn.Module):

    def __init__(self,
                 n_vocab,
                 n_blocks=12,
                 n_heads=8,
                 embedding_size=128,
                 d_model=4096,
                 d_ff=2048,
                 max_seq_len=256,
                 dropout=0.1,
                 padding_idx=None):
        super(Bertrando, self).__init__()
        self.n_vocab = n_vocab
        self.transformer_embeddings = TransformerEmbedding(n_vocab, embedding_size,
                                                           dropout=dropout,
                                                           padding_idx=padding_idx)
        self.embedding_hidden_fc = nn.Linear(embedding_size, d_model)

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
        x = self.embedding_hidden_fc(x)

        for block in self.transformer_blocks:
            x = block(x)

        return x
