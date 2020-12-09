import re
import fire


def _contains_alphabet(line):
    return re.search('[a-zA-Z]', line) is not None


def convert_word_per_line(dataset_path, output_path, min_line_length=3):
    with open(output_path, 'w') as trg:
        with open(dataset_path, 'r') as src:
            acc = []

            for line in src:
                line = line[:-1]
                if len(line) == 0 or line == '<s>':
                    sentence = ' '.join(acc)
                    acc = []
                    if len(sentence) >= min_line_length and _contains_alphabet(sentence):
                        trg.write(f'{sentence}\n')
                else:
                    acc.append(line)


if __name__ == '__main__':
    fire.Fire()
