"""
Test Dilated Causal Convolution
"""

import os
import sys

import torch
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wavenet.networks import CausalConv1d, DilatedCausalConv1d


CAUSAL_RESULT = [
    [[[18, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94]]]
]

DILATED_CAUSAL_RESULT = [
    [[[56,   80,  88,  96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184]]],
    [[[144, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]]],
    [[[368, 416, 448, 480, 512, 544, 576, 608, 640]]],
    [[[1008]]]
]


def causal_conv(data, in_channels, out_channels, print_result=True):
    conv = CausalConv1d(in_channels, out_channels)
    conv.init_weights_for_test()

    output = conv(data)

    print('Causal convolution ---')
    if print_result:
        print('    {0}'.format(output.data.numpy().astype(int)))

    return output


def dilated_causal_conv(step, data, channels, dilation=1, print_result=True):
    conv = DilatedCausalConv1d(channels, dilation=dilation)
    conv.init_weights_for_test()

    output = conv(data)

    print('{0} step is OK: dilation={1}, size={2}'.format(step, dilation, output.shape))
    if print_result:
        print('    {0}'.format(output.data.numpy().astype(int)))

    return output


@pytest.fixture
def generate_x():
    """Test normal convolution 1d"""
    x = np.arange(1, 33, dtype=np.float32)
    x = np.reshape(x, [1, 2, 16])  # [batch, channel, timestep]
    x = torch.autograd.Variable(torch.from_numpy(x))

    print('Input size={0}'.format(x.shape))
    print(x.data.numpy().astype(int))
    print('-'*80)

    return x


@pytest.fixture
def test_causal_conv(generate_x):
    """Test normal convolution 1d"""
    result = causal_conv(generate_x, 2, 1)

    np.testing.assert_array_equal(
        result.data.numpy().astype(int),
        CAUSAL_RESULT[0]
    )

    return result


def test_dilated_causal_conv(test_causal_conv):
    """Test dilated causal convolution : dilation=[1, 2, 4, 8]"""
    result = test_causal_conv

    for i in range(0, 4):
        result = dilated_causal_conv(i+1, result, 1, dilation=2**i)

        np.testing.assert_array_equal(
            result.data.numpy().astype(int),
            DILATED_CAUSAL_RESULT[i]
        )

