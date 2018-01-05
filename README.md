# WaveNet

Yet another WaveNet implementation in PyTorch.

The purpose of this implementation is Well-structured, reusable and easily understandable.

- [WaveNet Paper](https://arxiv.org/pdf/1609.03499.pdf)
- [WaveNet: A Generative Model for Raw Audio](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

## Prerequisites

- System
    - Linux or macOS
    - CPU or (NVIDIA GPU + CUDA CuDNN)
        - It can run on Single CPU/GPU or Multi GPUs.
    - Python 3

- Libraries
    - PyTorch >= 0.3.0
    - librosa >= 0.5.1

## Training

```bash
python train.py \
    --data_dir=./test/data \
    --output_dir=./outputs
```

Use `python train.py --help` to see more options.

## Generating

It's just for testing. You need to modify for real world.

```bash
python generate.py \
    --model=./outputs/model \
    --seed=./test/data/helloworld.wav \
    --out=./output/helloworld.wav
```

Use `python generate.py --help` to see more options.

## File structures

`networks.py` and `model.py` is main implementations.

- wavenet
    - `config.py` : Training options
    - `networks.py` : The neural network architecture of WaveNet
    - `model.py` : Calculate loss and optimizing
    - utils
        - `data.py` : Utilities for loading data
    - test
        - Some tests for check if it's correct model like casual, dilated..
- `train.py` : A script for WaveNet training
- `generate.py` : A script for generating with pre-trained model

# TODO

- [ ] Add some nice samples
- [ ] Global conditions
- [ ] Local conditions
- [ ] Faster generating
- [ ] Parallel WaveNet
- [ ] General Generator

## References

- https://github.com/ibab/tensorflow-wavenet
- https://qiita.com/MasaEguchi/items/cd5f7e9735a120f27e2a
- https://github.com/musyoku/wavenet/issues/4
- https://github.com/vincentherrmann/pytorch-wavenet
- http://sergeiturukin.com/2017/03/02/wavenet.html

