"""
Main model of WaveNet
Calculate loss and optimizing
"""
import os

import torch
import torch.optim

from wavenet.networks import WaveNet as WaveNetModule


class WaveNet:
    def __init__(self, layer_size, stack_size, in_channels, res_channels, lr=0.002):

        self.net = WaveNetModule(layer_size, stack_size, in_channels, res_channels)

        self.in_channels = in_channels
        self.receptive_fields = self.net.receptive_fields

        self.lr = lr
        self.loss = self._loss()
        self.optimizer = self._optimizer()

        self._prepare_for_gpu()

    @staticmethod
    def _loss():
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print("{0} GPUs are detected.".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net)

        if torch.cuda.is_available():
            self.net.cuda()

    def decay_optimizer(self, num_steps, decay_step):
        self.optimizer.param_groups[0]['lr'] -= self.lr / (num_steps - decay_step)

    def train(self, inputs, targets):
        """
        Train 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :param targets: Torch tensor [batch, timestep, channels]
        :return: float loss
        """
        outputs = self.net(inputs)

        loss = self.loss(outputs.view(-1, self.in_channels),
                         targets.long().view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def generate(self, inputs):
        """
        Generate 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        outputs = self.net(inputs)

        return outputs

    def load(self, model_dir):
        """
        Load pre-trained model
        :param model_dir:
        :return:
        """
        print("Loading model from {0}".format(model_dir))

        self.net.load_state_dict(torch.load(
                                 os.path.join(model_dir, 'wavenet.pkl')))

    def save(self, model_dir):
        print("Saving model into {0}".format(model_dir))

        torch.save(self.net.state_dict(),
                   os.path.join(model_dir, 'wavenet.pkl'))

