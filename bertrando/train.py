import os
import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from networks.mlm import MLMPreTraining
from networks.bertrando import Bertrando
from datasets.mlm_dataset import LineByLineTextDataset, DataCollatorForMLM


def load_tokenizer(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.post_processor = TemplateProcessing(
        single='[CLS] $A [SEP]',
        pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[('[CLS]', 1), ('[SEP]', 2)],
    )
    return tokenizer


def load_dataset(cfg, tokenizer):
    dataset = LineByLineTextDataset(to_absolute_path(cfg.dataset_path), tokenizer)
    val_dataset = LineByLineTextDataset(to_absolute_path(cfg.val_dataset_path), tokenizer)
    return dataset, val_dataset


def get_data_loaders(cfg, tokenizer):
    train, val = load_dataset(cfg, tokenizer)
    data_collator = DataCollatorForMLM(tokenizer, max_seq_len=cfg.max_seq_len)
    train_loader = DataLoader(
        train,
        batch_size=cfg.train_batch_size,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        val,
        batch_size=cfg.val_batch_size,
        collate_fn=data_collator
    )
    return train_loader, val_loader


def load_comet_logger(cfg):
    return CometLogger(api_key=os.getenv('COMET_API_KEY'),
                       project_name=os.getenv('COMET_PROJECTNAME'),
                       workspace=os.getenv('COMET_WORKSPACE'),
                       save_dir=to_absolute_path(cfg.logs.path))


def find_lr(trainer, model, dataloader):
    lr_finder = trainer.tuner.lr_find(model, train_dataloader=dataloader)
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_find_fig.png')
    print('Suggested_lr:', lr_finder.suggestion())


@hydra.main(config_path='../configs/', config_name='bertrando_small')
def train(cfg):
    tokenizer = load_tokenizer(to_absolute_path(cfg.tokenizer_path))
    train_loader, val_loader = get_data_loaders(cfg, tokenizer)
    max_steps = cfg.max_epochs * len(train_loader)

    backbone = Bertrando(
        tokenizer.get_vocab_size(),
        n_blocks=cfg.model.n_blocks,
        n_heads=cfg.model.n_heads,
        embedding_size=cfg.model.embedding_size,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.model.dropout,
        padding_idx=tokenizer.token_to_id('[PAD]')
    )
    model = MLMPreTraining(backbone, tokenizer.get_vocab_size(), cfg.model.d_model, n_training_steps=max_steps)

    comet_logger = load_comet_logger(cfg) if cfg.logs.use_comet else None
    logger = comet_logger or TensorBoardLogger(to_absolute_path(cfg.logs.path))
    logger.log_hyperparams(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        filename=f'{model}' + '-{epoch:02d}-{step}-{val_mlm_loss:.3f}',
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.checkpoint.mode
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.early_stopping_monitor,
        min_delta=cfg.early_stopping_delta,
        patience=cfg.early_stopping_patience,
        mode=cfg.early_stopping_mode
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        val_check_interval=cfg.val_check_interval,
        num_sanity_val_steps=-1,
        gpus=cfg.gpus,
        precision=cfg.precision,
        terminate_on_nan=True,
        logger=logger,
        log_every_n_steps=cfg.logs.log_every_n_steps,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    # find_lr(trainer, model, train_loader)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    load_dotenv()
    train()
