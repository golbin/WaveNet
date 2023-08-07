"""
A script for WaveNet training
"""
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wavenet.config as config
from wavenet.model import WaveNet
from wavenet.utils.data import DataLoader, Dataset, one_hot_decode


class Trainer:
    def __init__(self, args):
        self.args = args

        self.wavenet = WaveNet(args.layer_size, args.stack_size,
                               args.in_channels, args.res_channels,
                               lr=args.lr)

        self.data_loader = DataLoader(args.train_data_dir, self.wavenet.receptive_fields,
                                      args.sample_size, args.sample_rate, args.in_channels)

    def infinite_train_batch(self):
        while True:
            for dataset in self.data_loader:
                for inputs, targets in dataset:
                    yield inputs, targets

    def run(self):
        total_steps = 0

        for inputs, targets in self.infinite_train_batch():
            loss = self.wavenet.train(inputs, targets)

            total_steps += 1

            print('[{0}/{1}] loss: {2}'.format(total_steps, args.num_steps, loss))

            if total_steps > self.args.num_steps:
                break

        self.wavenet.save(args.model_dir)


class QuasiTrainer:
    def __init__(self, args):
        self.args = args

        self.wavenet = WaveNet(args.layer_size, args.stack_size,
                               args.in_channels, args.res_channels,
                               lr=args.lr)

        self.train_dataset = Dataset(
            args.train_data_dir, sample_rate=args.sample_rate, in_channels=args.in_channels, trim=False,
            use_shuffle=True
        )
        self.val_dataset = Dataset(
            args.val_data_dir, sample_rate=args.sample_rate, in_channels=args.in_channels, trim=False,
            use_shuffle=True
        )

        self.receptive_field = self.wavenet.receptive_fields
        self._sample_size = self.args.sample_size
        self._batch_size = self.args.batch_size

        self._writer = SummaryWriter(args.log_dir)

    def run(self):
        for epoch_idx in range(self.args.epochs):
            print(f"Epoch {epoch_idx}")
            train_loss = self._epoch(epoch_idx, "train")
            # save model after each epoch
            val_loss = self._epoch(epoch_idx, "val")
            print(f"Epoch: {epoch_idx}. Train loss: {train_loss}, val loss: {val_loss}")
            self.train_dataset.shuffle_set()
            self.val_dataset.shuffle_set()
            self.wavenet.save(args.model_dir, step=epoch_idx)

    @staticmethod
    def _calc_sample_size(sample_size, batch_size, receptive_fields):
        ss = sample_size * batch_size - receptive_fields * (batch_size - 1)
        return ss

    @staticmethod
    def _create_batch(inputs, batch_size, sample_size, receptive_fields):
        out = np.zeros([batch_size, sample_size, inputs.shape[-1]])
        for in_bs_pos in range(batch_size):
            audio_pos = in_bs_pos * (sample_size - receptive_fields)
            out[in_bs_pos, :, :] = inputs[:, audio_pos: audio_pos + sample_size, :]
        return out

    def _epoch(self, epoch_idx, mode="train"):  # train or val
        _set = self.train_dataset if mode == "train" else self.val_dataset
        _log_cat = "Loss/train" if mode == "train" else "Loss/val"
        steps = 0
        losses = []
        for audio in tqdm(_set):
            audio = np.expand_dims(audio, 0)
            audio = np.pad(audio, [[0, 0], [self.receptive_field, 0], [0, 0]], 'constant')

            sample_size = self._calc_sample_size(
                self._sample_size, self._batch_size, self.wavenet.receptive_fields
            )
            while audio.shape[1] > sample_size:
                inputs = audio[:, :sample_size, :]

                batched_inputs = self._create_batch(inputs, self._batch_size, self._sample_size, self.receptive_field)
                batched_targets = batched_inputs[:, self.receptive_field:, :]

                batched_inputs = DataLoader._variable(batched_inputs)
                batched_targets = DataLoader._variable(one_hot_decode(batched_targets, 2))

                if mode == "train":
                    loss = self.wavenet.train(batched_inputs, batched_targets)
                else:  # evaluate
                    loss = self.wavenet.eval(batched_inputs, batched_targets)
                losses.append(loss)

                audio = audio[:, sample_size - self.receptive_field:, :]
                sample_size = self._calc_sample_size(
                    self._sample_size, self._batch_size, self.wavenet.receptive_fields
                )

                steps += 1
        _loss = np.mean(np.array(losses))
        self._writer.add_scalar(_log_cat, _loss, epoch_idx)
        return _loss


def prepare_output_dir(args):
    args.log_dir = os.path.join(args.output_dir, 'log')
    args.model_dir = os.path.join(args.output_dir, 'model')
    args.test_output_dir = os.path.join(args.output_dir, 'test')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)


if __name__ == '__main__':
    args = config.parse_args()

    prepare_output_dir(args)

    if args.quasi:
        trainer = QuasiTrainer(args)
    else:
        trainer = Trainer(args)

    trainer.run()
