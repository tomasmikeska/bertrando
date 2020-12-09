import fire
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, Strip


def train(dataset_path,
          output_dir='data/tokenizer/',
          vocab_size=30_000,
          min_frequency=3):

    trainer = BpeTrainer(vocab_size=vocab_size,
                         min_frequency=min_frequency,
                         special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'])
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Sequence([Lowercase(), Strip()])

    files = [dataset_path]
    tokenizer.train(trainer, files)

    files = tokenizer.model.save(output_dir)
    tokenizer.model = BPE.from_file(*files, unk_token='[UNK]')

    tokenizer.save(f'{output_dir}/tokenizer.json')


if __name__ == '__main__':
    fire.Fire(train)