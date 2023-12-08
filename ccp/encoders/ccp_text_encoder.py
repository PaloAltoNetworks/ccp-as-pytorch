from typing import Tuple

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(
        self,
        dim_in: Tuple[int, int],
        conv1a_num_filters=32,
        conv1a_size=5,
        conv1b_num_filters=16,
        conv1b_size=3,
        conv2a_num_filters=32,
        conv2a_size=9,
        conv2b_num_filters=16,
        conv2b_size=7,
    ):
        """
        Encoder f_b(embedding_X) = b. Defaults are the configuration from paper.

        We have parallel convolutional layers:
        (1) one layer of 32 filters each of size 5x100, passed into 16 filters
            each of size 3x1, into a global max pooling layer
        (2) one layer of 32 filters each of size 9x100, passed into 16 filters
            each of size 7x1, into a global max pooling layer
        We concatenate all the max pooling layers to form the output `b`.

        Parameters:
            dim_in: tuple representing the dimensionality of the input, as (text
                sequence length, dimensionality of embedding); we assume there is only
                one channel of input

        """
        super(TextEncoder, self).__init__()
        self.X_length, self.X_width = dim_in

        self.conv1a_num_filters = conv1a_num_filters
        self.conv1a_size = conv1a_size
        self.conv1b_num_filters = conv1b_num_filters
        self.conv1b_size = conv1b_size

        self.conv2a_num_filters = conv2a_num_filters
        self.conv2a_size = conv2a_size
        self.conv2b_num_filters = conv2b_num_filters
        self.conv2b_size = conv2b_size

        self.model_a = nn.Sequential(
            # output is [batch_size, self.conv1b_num_filters, 1, 1]
            nn.Conv2d(
                in_channels=1,
                out_channels=self.conv1a_num_filters,
                kernel_size=(self.conv1a_size, self.X_width),
            ),
            nn.Conv2d(
                in_channels=self.conv1a_num_filters,
                out_channels=self.conv1b_num_filters,
                kernel_size=(self.conv1b_size, 1),
            ),
            nn.MaxPool2d(
                kernel_size=(
                    self.X_length - self.conv1a_size + 1 - self.conv1b_size + 1,
                    1,
                )
            ),
        )

        self.model_b = nn.Sequential(
            # output is [batch_size, self.conv2b_num_filters, 1, 1]
            nn.Conv2d(
                in_channels=1,
                out_channels=self.conv2a_num_filters,
                kernel_size=(self.conv2a_size, self.X_width),
            ),
            nn.Conv2d(
                in_channels=self.conv2a_num_filters,
                out_channels=self.conv2b_num_filters,
                kernel_size=(self.conv2b_size, 1),
            ),
            nn.MaxPool2d(
                kernel_size=(
                    self.X_length - self.conv2a_size + 1 - self.conv2b_size + 1,
                    1,
                )
            ),
        )

    def forward(self, X):
        """
        Produces the convolutionally encoded `b`.

        Input shape (4D):
            BATCH_SIZE x 1 channel x self.X_length x self.X_width
        Output shape (2D):
            BATCH_SIZE x (self.conv1b_num_filters + self.conv2b_num_filters)
        """
        if X.shape[1:] != torch.Size([1, self.X_length, self.X_width]):
            raise ValueError(
                f"Unexpected text embedding shape {X.shape}; expectation is (batch_size x 1 channel x *dim_in) = (batch_size x 1 x {self.X_length} x {self.X_width})"
            )

        out = torch.squeeze(torch.cat([self.model_a(X), self.model_b(X)], dim=1))
        return out

    @property
    def output_dim(self) -> int:
        """
        Returns the number of output dimensions. After encoding, each element
        of the batch has `output_dim` size.
        """
        return self.conv1b_num_filters + self.conv2b_num_filters
