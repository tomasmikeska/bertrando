# Bertrando

Experiments with efficient Transformers.

### Requirements

- Docker
- nvidia-docker

### Installation

Build Docker image
```
$ docker build -t bertrando .
```

Run nvidia-docker image
```
$ docker run -it -v $(pwd):/bertrando --gpus all --ipc=host --runtime=nvidia bertrando bash
```

### Usage

Training pipeline is configured using Hydra config files present in `configs/`. All options in config file
can be overwritten using command-line arguments. (Hydra docs: https://hydra.cc/docs/intro)

Train tokenizer
```
$ python bertrando/tokenizers/wordpiece.py --dataset_path data/train.txt
```

Train model
```
$ python bertrando/train.py
```
