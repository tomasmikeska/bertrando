import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformer_layer import TransformerEncoderLayer
from positional_encoding import LearnedPositionalEmbedding
from optimization import get_linear_schedule_with_warmup


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
        self.token_embedding = nn.Embedding(n_vocab, d_model, padding_idx=padding_idx)
        self.positional_embedding = LearnedPositionalEmbedding(d_model)
        # self.segment_encoding = nn.Embedding(2, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(p=dropout)

        self.transformer_blocks = nn.ModuleList()
        for i in range(n_blocks):
            block = TransformerEncoderLayer(
                d_model,
                n_heads,
                d_ff,
                dropout
            )
            self.transformer_blocks.append(block)

    def forward(self, inputs, segments):
        x = self.token_embedding(inputs)
        x += self.positional_embedding(inputs)
        # x += self.segment_encoding(segments)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        return x


class MLMPreTraining(pl.LightningModule):

    def __init__(self, model, n_vocab, d_model, learning_rate=1e-4, n_warmup_steps=1e4, n_training_steps=1e6):
        super(MLMPreTraining, self).__init__()
        self.model = model
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps

        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_vocab, bias=False)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, segment):
        return self.model(input, segment)

    def training_step(self, batch, batch_idx):
        original_input = batch['original_input']
        masked_input   = batch['masked_input']
        mask           = batch['mask']
        segment        = batch['segment']

        x = self.forward(masked_input, segment)

        masked_x = x.masked_select(mask.unsqueeze(-1)).view(-1, self.d_model)
        y = self.cls(masked_x)

        mlm_loss = self.loss_fn(y.view(-1, self.n_vocab), original_input.masked_select(mask).view(-1))

        self.log('train_mlm_loss', mlm_loss)

        return mlm_loss

    def validation_step(self, batch, batch_idx):
        original_input = batch['original_input']
        masked_input   = batch['masked_input']
        mask           = batch['mask']
        segment        = batch['segment']

        x = self.forward(masked_input, segment)

        masked_x = x.masked_select(mask.unsqueeze(-1)).view(-1, self.d_model)
        y = self.cls(masked_x)

        mlm_loss = self.loss_fn(y.view(-1, self.n_vocab), original_input.masked_select(mask).view(-1))

        self.log('val_mlm_loss', mlm_loss)

        return mlm_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            self.n_warmup_steps,
            self.n_training_steps
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def __str__(self):
        return 'bert-mlm'
