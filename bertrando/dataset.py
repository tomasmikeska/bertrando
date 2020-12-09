import torch
from torch.utils.data import IterableDataset


class LineByLineTextDataset(IterableDataset):

    def __init__(self, dataset_path, tokenizer, min_line_length=2):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.dataset_len = self._compute_num_lines()
        self.min_line_length = min_line_length

    def _is_line_accepted(self, line):
        return len(line.split()) >= self.min_line_length

    def _compute_num_lines(self):
        count = 0
        with open(self.dataset_path) as f:
            for line in f:
                if self._is_line_accepted(line):
                    count += 1
        return count

    def _process_line(self, line):
        return self.tokenizer.encode(line)

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        file_iter = open(self.dataset_path)
        file_iter = filter(self._is_line_accepted, file_iter)
        return map(lambda l: self.tokenizer.encode(l), file_iter)


class DataCollatorForMLM:

    def __init__(self,
                 tokenizer,
                 mlm_probability=0.15,
                 max_seq_len=256,
                 mask_token='[MASK]',
                 n_reserved_tokens=5):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_seq_len = max_seq_len
        self.mask_token = mask_token
        self.n_reserved_tokens = n_reserved_tokens

        self.batch = None

    def __call__(self, examples):
        self._pad(examples)
        batch = self._batchify(examples)
        return self._mask_tokens(batch)

    def _pad(self, examples):
        longest_line = max([len(l) for l in examples])
        for example in examples:
            example.pad(longest_line)
            example.truncate(self.max_seq_len)

    def _batchify(self, examples):
        return {
            'original_input': torch.stack([torch.tensor(x.ids) for x in examples]),
            'segment': torch.stack([torch.tensor(x.type_ids) for x in examples]),
            'special_tokens_mask': torch.stack([torch.tensor(x.special_tokens_mask) for x in examples])
        }

    def _mask_tokens(self, batch):
        '''
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        '''
        original_tokens     = batch['original_input']
        special_tokens_mask = batch['special_tokens_mask'].bool()

        masked_input = original_tokens.clone()
        labels = original_tokens.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(masked_input.shape, self.mlm_probability)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[(~masked_indices)] = -100

        # 80% of the time, replace masked input tokens with mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(masked_input.shape, 0.8)).bool() & masked_indices
        masked_input[indices_replaced] = self.tokenizer.token_to_id(self.mask_token)

        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(masked_input.shape, 0.5)).bool()
        indices_random = indices_random & masked_indices & ~indices_replaced
        random_words = torch.randint(self.n_reserved_tokens,  # Don't use special tokens as random words
                                     self.tokenizer.get_vocab_size(),
                                     masked_input.shape,
                                     dtype=torch.long)
        masked_input[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return {**batch, 'masked_input': masked_input, 'mask': masked_indices}
