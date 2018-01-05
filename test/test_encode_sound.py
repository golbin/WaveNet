"""
Test mu-law encoding and decoding
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wavenet.utils.data import *


SAMPLE_RATE = 8000
QUANTIZATION_CHANNEL = 256

TEST_AUDIO_FILE = os.path.join(os.path.dirname(__file__),
                               'data', 'helloworld.wav')


def test_mu_law_encode():
    raw_audio = load_audio(TEST_AUDIO_FILE, SAMPLE_RATE)
    raw_audio = raw_audio[2007:2013, :]

    mu_law_encoded = mu_law_encode(raw_audio, QUANTIZATION_CHANNEL)
    mu_law_decoded = mu_law_decode(mu_law_encoded, QUANTIZATION_CHANNEL)
    one_hot_encoded = one_hot_encode(mu_law_encoded, QUANTIZATION_CHANNEL)
    one_hot_decoded = one_hot_decode(one_hot_encoded)
    one_hot_decoded.shape = (one_hot_decoded.size, 1)

    print('--- Raw audio ---')
    print(raw_audio)
    print('--- mu-law encoded ---')
    print(mu_law_encoded)
    print('--- mu-law decoded ---')
    print(mu_law_decoded)
    print('--- one-hot encoded ---')
    print(one_hot_encoded)
    print('--- one-hot decoded ---')
    print(one_hot_decoded)

    np.testing.assert_array_equal(mu_law_encoded, one_hot_decoded)

    assert np.min(raw_audio - mu_law_decoded) < 0.0011
    assert np.max(raw_audio - mu_law_decoded) < 0.012
    assert np.mean(raw_audio - mu_law_decoded) < 0.0036

