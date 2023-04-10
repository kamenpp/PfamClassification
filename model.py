import torch
import torch.nn as nn
from resnet_block import ResNetBlock


class Model(nn.Module):
    """
    Protein sequence classification model.

    Conv1d (1x1) -> ResNet block -> Pooling (3 x 2) -> Dense -> Softmax

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        input_length (int): Length of input sequence.
        out_features (int): Number of output features.
        pooling_type (str): Type of pooling used. Can be "max" or "avg".
        kernel_size_of_first_resnet_conv (int): Kernel size of the first convolutional layer in the ResNet block.
        dilation_of_first_resnet_conv (int): Dilation of the first convolutional layer in the ResNet block.
        stride_of_first_resnet (int): Stride of the first convolutional layer in the ResNet block.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_length: int,
                 out_features: int,
                 kernel_size_of_first_resnet_conv: int = 7,
                 dilation_of_first_resnet_conv: int = 3,
                 stride_of_first_resnet: int = 1,
                 pooling_type: str = 'max',
                 pooling_kernel_size: int = 3,
                 pooling_stride: int = 2
                 ):

        super(Model, self).__init__()

        assert pooling_type in ['max', 'avg'], 'The pooling type must be "avg" or "max".'

        # Input convolution (1x1) -> To capture the most basic input features
        self.input_conv1d = nn.Conv1d(in_channels,
                                      out_channels,
                                      dilation=1,
                                      kernel_size=1,
                                      padding='same',
                                      bias=False)

        # ResNet block
        self.resnet1 = ResNetBlock(out_channels,
                                   out_channels,
                                   kernel_size_of_first_conv=kernel_size_of_first_resnet_conv,
                                   dilation_of_first_conv=dilation_of_first_resnet_conv,
                                   stride_of_first_conv=stride_of_first_resnet)

        # Pooling layer
        if pooling_type == 'max':
            self.pool1d = nn.MaxPool1d(pooling_kernel_size, stride=pooling_stride, padding=0)
        elif pooling_type == 'avg':
            self.pool1d = nn.AvgPool1d(pooling_kernel_size, stride=pooling_stride, padding=0)

        # work out the shape for the linear layer,
        # as in https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        out_features_after_max_pooling = 1 + (input_length + 2 * self.pool1d.padding -
                                              self.pool1d.dilation * (
                                                          self.pool1d.kernel_size - 1) - 1) // self.pool1d.stride

        # Linear layer
        self.linear = nn.Linear(in_features=out_features_after_max_pooling * out_channels, out_features=out_features)

        # Softmax along the sequence dimension;
        # shape of input is (N, L_out), where N is the batch size and L_out is the output features
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (N, C, L_in).
            where N is the batch size, C is the input channels,
            L_in is the length of the input sequence (after padding, for example, 308)

        Returns:
            torch.Tensor: Output tensor of shape (N, L_out).
            where L_out is the output length which is
            the number of output classes (i.e., output features)
        """

        x = self.input_conv1d(x)
        x = self.resnet1(x)
        x = self.pool1d(x)

        # Flatten the output and pass through the linear layer
        x = self.linear(x.view(x.shape[0], -1))

        return self.softmax(x)
