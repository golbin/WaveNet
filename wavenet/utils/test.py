"""
Utilities for testing
"""


def accuracy(predict, target):
    """
    Calculate accuracy
    :param predict: Tensor
    :param target:  Tensor
    :return:
    """
    acc_val = (predict.cpu().data.numpy() == target.cpu().data.numpy()).mean()

    return acc_val

