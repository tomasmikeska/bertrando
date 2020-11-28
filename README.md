# Bumblebee Transformer

Experiments with efficient Transformers.

### Requirements

- Docker
- nvidia-docker

### Installation

Build Docker image
```
$ docker build -t bumblebee .
```

Run nvidia-docker image
```
$ docker run -it -v $(pwd):/bumblebee --gpus all --ipc=host --runtime=nvidia bumblebee bash
```

### Usage

Training pipeline is configured using Hydra config files present in `configs/`. All options in config file
can be overwritten using command-line arguments. (Hydra docs: https://hydra.cc/docs/intro)

Train model
```
$ python src/train.py
```
