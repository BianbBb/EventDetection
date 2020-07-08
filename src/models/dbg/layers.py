import torch.nn as nn


def conv1d(in_channels, out_channels, kernel_size=3, is_relu=True):
    """
    Construct Conv1D operation
    :param in_channels: channel number of input tensor
    :param out_channels: channel number of output tensor
    :param kernel_size: int
    :param is_relu: bool, use ReLU or not
    :return: Conv1D module
    """
    if is_relu:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2)
        )


def conv2d(in_channels, out_channels, kernel_size=3, is_relu=True):
    """
    Construct Conv2D operation
    :param in_channels: channel number of input tensor
    :param out_channels: channel number of output tensor
    :param kernel_size: int
    :param is_relu: bool, use ReLU or not
    :return: Conv2D module
    """
    if is_relu:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) // 2)
        )
