import torch
import pytorch_lightning as pl
from layers.transformer_head import MLMLossHead
from optimization import get_linear_schedule_with_warmup


class MLMPreTraining(pl.LightningModule):

    def __init__(self,
                 model,
                 n_vocab,
                 d_model,
                 learning_rate=1e-4,
                 n_warmup_steps=1e4,
                 n_training_steps=1e6):
        super(MLMPreTraining, self).__init__()
        self.model = model
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps

        self.loss_head = MLMLossHead(d_model, n_vocab)

    def forward(self, input, segment=None):
        return self.model(input, segment)

    def training_step(self, batch, batch_idx):
        original_input = batch['original_input']
        masked_input   = batch['masked_input']
        mask           = batch['mask']

        x = self.forward(masked_input)

        mlm_loss = self.loss_head(x, original_input, mask)

        self.log('train_mlm_loss', mlm_loss)

        return mlm_loss

    def validation_step(self, batch, batch_idx):
        original_input = batch['original_input']
        masked_input   = batch['masked_input']
        mask           = batch['mask']

        x = self.forward(masked_input)

        mlm_loss = self.loss_head(x, original_input, mask)

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
