"""
A script for WaveNet training
"""
import torch
import librosa
import datetime
import numpy as np

import wavenet.config as config
from wavenet.model import WaveNet
import wavenet.utils.data as utils


class Generator:
    def __init__(self, args):
        self.args = args

        self.wavenet = WaveNet(args.layer_size, args.stack_size,
                               args.in_channels, args.res_channels)

        self.wavenet.load(args.model_dir, args.step)

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    def _make_seed(self, audio):
        audio = np.pad([audio], [[0, 0], [self.wavenet.receptive_fields, 0], [0, 0]], 'constant')

        if self.args.sample_size:
            seed = audio[:, :self.args.sample_size, :]
        else:
            seed = audio[:, :self.wavenet.receptive_fields*2, :]

        return seed

    def _get_seed_from_audio(self, filepath):
        audio = utils.load_audio(filepath, self.args.sample_rate)
        audio_length = len(audio)

        audio = utils.mu_law_encode(audio, self.args.in_channels)
        audio = utils.one_hot_encode(audio, self.args.in_channels)

        seed = self._make_seed(audio)

        return self._variable(seed), audio_length

    def _save_to_audio_file(self, data):
        data = data[0].cpu().data.numpy()
        data = utils.one_hot_decode(data, axis=1)
        audio = utils.mu_law_decode(data, self.args.in_channels)

        librosa.output.write_wav(self.args.out, audio, self.args.sample_rate)
        print('Saved wav file at {}'.format(self.args.out))

        return librosa.get_duration(y=audio, sr=self.args.sample_rate)

    def generate(self):
        outputs = []
        inputs, audio_length = self._get_seed_from_audio(self.args.seed)

        while True:
            new = self.wavenet.generate(inputs)

            outputs = torch.cat((outputs, new), dim=1) if len(outputs) else new

            print('{0}/{1} samples are generated.'.format(len(outputs[0]), audio_length))

            if len(outputs[0]) >= audio_length:
                break

            inputs = torch.cat((inputs[:, :-len(new[0]), :], new), dim=1)

        outputs = outputs[:, :audio_length, :]

        return self._save_to_audio_file(outputs)


if __name__ == '__main__':
    args = config.parse_args(is_training=False)

    generator = Generator(args)

    start_time = datetime.datetime.now()

    duration = generator.generate()

    print('Generate {0} seconds took {1}'.format(duration, datetime.datetime.now() - start_time))

