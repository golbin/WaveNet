from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd

from wavenet.model import WaveNet
from wavenet.utils.data import DataLoader


def get_config():
    return {
        "net_layer_size": [10],
        "net_stack_size": [3, 4, 5],
        "net_in_channels": [256],
        "net_res_channels": [512],
        "batch_size": [1, 2, 4, 8, 16],
        "samples_per_batch": [5 * 1024, 6 * 1024, 7 * 1024, 8 * 1024, 9 * 1024, 10 * 1024],
        "trials": 5000,
        "output_path": Path("./")
    }


def _net_param_generator(config):
    for layer_size in config["net_layer_size"]:
        for stack_size in config["net_stack_size"]:
            for in_channels in config["net_in_channels"]:
                for res_channels in config["net_res_channels"]:
                    yield layer_size, stack_size, in_channels, res_channels


def _data_param_generator(config):
    for batch_size in config["batch_size"]:
        for samples_per_batch in config["samples_per_batch"]:
            yield batch_size, samples_per_batch


def perform_test(config):
    # per net config
    for layer_size, stack_size, in_channels, res_channels in _net_param_generator(config):
        wavenet = WaveNet(layer_size, stack_size, in_channels, res_channels, 1e-3)
        net_data_results = defaultdict(list)
        for batch_size, samples_per_batch in _data_param_generator(config):
            with_data = np.zeros(config["trials"], dtype=np.float32)
            without_data = np.zeros(config["trials"], dtype=np.float32)
            samples_per_datapoint = samples_per_batch / batch_size
            for trial_idx in range(config["trials"]):
                # create data
                _input = np.zeros(
                    [batch_size, samples_per_datapoint, in_channels],
                    dtype=np.float32
                )
                _target = np.random.randint(
                    low=0, high=in_channels, size=[batch_size, samples_per_datapoint - wavenet.receptive_fields]
                )
                _input_ones = np.random.randint(low=0, high=in_channels, size=samples_per_datapoint)
                _input[:, np.arange(samples_per_datapoint), _input_ones] = 1.

                # send data to device
                data_send_start = time.time()
                dev_input = DataLoader._variable(_input)
                dev_target = DataLoader._variable(_target)
                # forward pass with backprop
                train_iteration_start = time.time()
                outputs = wavenet.net(dev_input)

                loss = wavenet.loss(outputs.view(-1, wavenet.in_channels),
                                    dev_target.long().view(-1))

                wavenet.optimizer.zero_grad()
                loss.backward()
                wavenet.optimizer.step()

                end_time = time.time()
                with_data[trial_idx] = end_time - data_send_start
                without_data[trial_idx] = end_time - train_iteration_start

            net_data_results["batch_size"].append(batch_size)
            net_data_results["samples_per_datapoint"].append(samples_per_datapoint)
            net_data_results["samples_per_batch"].append(samples_per_batch)
            net_data_results["with_data_mean"].append(np.mean(with_data))
            net_data_results["with_data_std"].append(np.std(with_data))
            net_data_results["without_data_mean"].append(np.mean(without_data))
            net_data_results["without_data_std"].append(np.std(without_data))

        net_data_results_df = pd.DataFrame.from_dict(net_data_results)
        outfilename = config["output_path"] / f"ls{layer_size}_ss{stack_size}_ic{in_channels}_rc{res_channels}.csv"
        net_data_results_df.to_csv(outfilename, index=False)
