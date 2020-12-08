from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, Strip


if __name__ == '__main__':
    trainer = BpeTrainer(vocab_size=30_000,
                         min_frequency=3,
                         special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'])
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Sequence([Lowercase(), Strip()])

    files = [f'data/{split}.txt' for split in ['train', 'val']]
    tokenizer.train(trainer, files)

    files = tokenizer.model.save('data/cs_tokenizer/')
    tokenizer.model = BPE.from_file(*files, unk_token='[UNK]')

    tokenizer.save('data/cs_tokenizer/tokenizer.json')
