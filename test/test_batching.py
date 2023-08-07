import numpy as np

from train import QuasiTrainer

_LAYER_SIZE = 5
_STACK_SIZE = 2
_IN_CHANNELS = 256
_RES_CHANNELS = 64
_UNIT = 64
_DATA_SIZE = 100 * _UNIT + 5  # to make it not dividable by 1024
SAMPLE_SIZES = [5 * _UNIT, 10 * _UNIT]
RECEPTIVE_FIELDS = [2 * _UNIT, 3 * _UNIT, 4 * _UNIT]
BATCH_SIZES = [1, 2, 4, 8]
DATA = np.random.uniform(0, 1, (1, _DATA_SIZE, _IN_CHANNELS))


def _old_input_target(sample_size, receptive_fields):
    input_out = []
    target_out = []
    temp_data = DATA
    while temp_data.shape[1] >= sample_size:
        input_out.append(
            temp_data[:, :sample_size, :]
        )
        target_out.append(
            temp_data[:, receptive_fields:sample_size, :]
        )
        temp_data = temp_data[:, sample_size - receptive_fields:, :]
    return np.concatenate(input_out, axis=0), np.concatenate(target_out, axis=0)


def _new_input_target(sample_size, batch_size, receptive_fields):
    input_out = []
    target_out = []
    temp_data = DATA
    calc_sample_size = QuasiTrainer._calc_sample_size(
        sample_size, batch_size, receptive_fields
    )
    while temp_data.shape[1] > calc_sample_size:
        inputs = temp_data[:, :calc_sample_size, :]
        batched_inputs = QuasiTrainer._create_batch(inputs, batch_size, sample_size, receptive_fields)
        batched_targets = batched_inputs[:, receptive_fields:, :]

        temp_data = temp_data[:, calc_sample_size - receptive_fields:, :]
        calc_sample_size = QuasiTrainer._calc_sample_size(
            sample_size, batch_size, receptive_fields
        )
        input_out.append(batched_inputs)
        target_out.append(batched_targets)
    return np.concatenate(input_out, axis=0), np.concatenate(target_out, axis=0)


def test_new_batching():
    for _ss in SAMPLE_SIZES:
        for _rf in RECEPTIVE_FIELDS:
            for _bs in BATCH_SIZES:
                old_input, old_target = _old_input_target(_ss, _rf)
                new_input, new_target = _new_input_target(_ss, _bs, _rf)
                old_input = old_input[:new_input.shape[0], :, :]
                old_target = old_target[:new_target.shape[0], :, :]

                assert np.array_equal(old_input, new_input)
                assert np.array_equal(old_target, new_target)
