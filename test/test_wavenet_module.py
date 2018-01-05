"""
Test mu-law encoding and decoding
"""

import os
import sys

import torch
import numpy as np

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wavenet.networks import WaveNet
from wavenet.exceptions import InputSizeError


LAYER_SIZE = 5  # 10 in paper
STACK_SIZE = 2  # 5 in paper
IN_CHANNELS = 2  # 256 in paper. quantized and one-hot input.
RES_CHANNELS = 512  # 512 in paper


def generate_dummy(dummy_length):
    x = np.arange(0, dummy_length, dtype=np.float32)
    x = np.reshape(x, [1, int(dummy_length / 2), 2])  # [batch, timestep, channels]
    x = torch.autograd.Variable(torch.from_numpy(x))

    return x


@pytest.fixture
def wavenet():
    net = WaveNet(LAYER_SIZE, STACK_SIZE, IN_CHANNELS, RES_CHANNELS)

    print(net)

    return net


def test_wavenet_output_size(wavenet):
    x = generate_dummy(wavenet.receptive_fields * 2 + 2)

    output = wavenet(x)

    # input size = receptive field size + 1 (* two channels)
    # output size = input size - receptive field size
    #             = 1
    assert output.shape == torch.Size([1, 1, 2])

    x = generate_dummy(wavenet.receptive_fields * 4)

    output = wavenet(x)

    # input size = receptive field size * 2 (* two channels)
    # output size = input size - receptive field size
    #             = receptive field size
    assert output.shape == torch.Size([1, wavenet.receptive_fields, 2])


def test_wavenet_fail_with_short_input(wavenet):
    x = generate_dummy(wavenet.receptive_fields * 2)

    try:
        wavenet(x)
        raise pytest.fail("Should be failed. Input size is too short.")
    except InputSizeError:
        pass

