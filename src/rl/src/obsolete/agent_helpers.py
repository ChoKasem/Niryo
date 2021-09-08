""" 
OS stands for output shape
"""

import torch.nn as nn
import numpy as np

def init_weights(m):
    """Weight initialization to be used by torch nn apply() function.
    HELPER FUNCTION to be used by the ActorCritic object.

    Args:
        m (torch.nn): Pytorch nn layer
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, nn.Conv2d):
        n_weights = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal_(m.weight, mean=0., std=np.sqrt(2. / n_weights))


def get_cu_OS(img_shape, cuPars, verbose=False):
    """Gets the output shape of the agent.ConvUnit output

    Args:
        img_shape (np.array): 3D input shape
        cuPars (ConvUnitPars): Parameters that define a ConvUnit.
        verbose (bool, optional): Whether or not to print the output shape of
            the convolution and maxpool as well. Defaults to False.

    Returns:
        list[int]: 3D shape of the ConvUnit output
    """
    def get_output_size_1D(input_size, kernel_size, padding, stride):
        """Gets the size of the filter output along one dimension

        Args:
            input_size (int): Size of input along the dimension
            kernel_size (int): Size of the filter along the dimension
            padding (int): Amount of padding along the dimension
            stride (int): Stride length along the dimension

        Returns:
            int: Size of the filter output along one dimension
        """
        return int(np.floor((input_size - kernel_size + 2*padding) / stride + 1))

    input_length, input_width, _ = img_shape

    # Through the conv layer
    output_length = get_output_size_1D(
        input_length, cuPars.conv_kernel, cuPars.conv_pad, cuPars.conv_stride
    )
    output_width = get_output_size_1D(
        input_width, cuPars.conv_kernel, cuPars.conv_pad, cuPars.conv_stride
    )
    if verbose:
        print("Conv Out", (output_length, output_width, cuPars.out_channels))

    # Through the maxpool layer
    output_length = get_output_size_1D(
        output_length, cuPars.maxpool_kernel, 0, cuPars.maxpool_stride
    )
    output_width = get_output_size_1D(
        output_width, cuPars.maxpool_kernel, 0, cuPars.maxpool_stride
    )
    if verbose:
        print("MaxPool Out", (output_length,
                              output_width, cuPars.out_channels))

    return [output_length, output_width, cuPars.out_channels]


class ConvUnitPars:
    def __init__(self, out_channels, conv_kernel, conv_stride, conv_pad, maxpool_kernel, maxpool_stride):
        self.out_channels = out_channels
        self.conv_kernel = conv_kernel  # Kernel Size
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad
        self.maxpool_kernel = maxpool_kernel  # Kernel Size
        self.maxpool_stride = maxpool_stride


class ConvUnit(nn.Module):
    def __init__(self, input_shape, cuPars):
        """This unit consists of a 2D convolution layer, a 2D maxpool layer,
        a 2D batchnorm layer, and ReLU activation.

        Args:
            input_shape (list[int]): Shape of the input,
                consisting of [height, width, num_channels]
            cuPars (ConvUnitPars): Pars of the ConvUnit
        """
        super(ConvUnit, self).__init__()

        _, _, num_channels = input_shape
        self.unit = nn.Sequential(
            nn.Conv2d(
                num_channels,
                cuPars.out_channels,
                kernel_size=cuPars.conv_kernel,
                stride=cuPars.conv_stride,
                padding=cuPars.conv_pad
            ),
            nn.MaxPool2d(
                cuPars.maxpool_kernel,
                stride=cuPars.maxpool_stride
            ),
            nn.BatchNorm2d(cuPars.out_channels),
            nn.ReLU()  # TODO: Check if this should be in every ConvUnit
        )

    def forward(self, x):
        return self.unit(x)