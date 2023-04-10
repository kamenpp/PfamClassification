import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """
    ResNetBlock is a building block used in Residual Neural Networks (ResNets). ResNets were proposed to solve the
    problem of vanishing gradients when training deep neural networks. The idea behind ResNets is to learn a residual
    mapping, i.e. the difference between the input and output of a block. This residual mapping is then added to the
    original input to produce the output. By doing this, the network is able to learn the identity function if
    needed, and can skip over unnecessary layers that might not be adding any value to the network.

    BtchNorm1d -> ReLU -> Conv1d -> BatchNorm1d -> ReLU -> Conv1d (bottleneck) -> skip connection

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size_of_first_conv (int): kernel size of the first convolutional layer (default: 7)
        dilation_of_first_conv (int): dilation of the first convolutional layer (default: 3)
        stride_of_first_conv (int): stride of the first convolutional layer (default: 1)

    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size_of_first_conv: int = 7,
                 dilation_of_first_conv: int = 3,
                 stride_of_first_conv: int = 1):
        super(ResNetBlock, self).__init__()

        # the first convolution is dilated by a specified amount
        # bias=False because a BatchNorm layer follows
        # this reduces the number of parameters in the model
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=kernel_size_of_first_conv,
                               stride=stride_of_first_conv,
                               padding='same',
                               dilation=dilation_of_first_conv,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # The second convolution is the bottleneck convolution,
        # dilation is always 1, and kernel size is always 1
        self.conv2 = nn.Conv1d(out_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               dilation=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNetBlock module.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: The output tensor with shape (batch_size, out_channels, input_length).
        """

        # save the residual for the skip connection
        residual = x

        x = self.bn1(x)
        x = self.relu(x)
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # skip connection element-wise addition
        # torch.add is not different from the normal '+' operator in this case
        out += residual

        return out
