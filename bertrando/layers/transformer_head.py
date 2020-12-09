import torch.nn as nn
from layers.layer_norm import LayerNorm


class MLMLossHead(nn.Module):

    def __init__(self, d_model, n_vocab):
        super(MLMLossHead, self).__init__()
        self.d_model = d_model
        self.n_vocab = n_vocab
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            LayerNorm(d_model),
            nn.Linear(d_model, n_vocab, bias=False)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, labels, mask):
        masked_x = x.masked_select(mask.unsqueeze(-1)).view(-1, self.d_model)
        y = self.cls(masked_x)

        return self.loss_fn(
            y.view(-1, self.n_vocab),
            labels.masked_select(mask).view(-1)
        )
