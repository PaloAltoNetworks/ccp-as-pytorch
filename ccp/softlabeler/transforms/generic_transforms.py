from typing import Tuple

import torch

from ccp.device_decision import DEVICE


class Identity(object):
    def __call__(self, tensor):
        return tensor


class GaussianNoise(object):
    def __init__(
        self,
        mean_range: Tuple[float, float] = (-0.5, 0.5),
        std_range: Tuple[float, float] = (0.01, 0.05),
    ):
        """
        Defines a Transform for adding random Gaussian Noise to every value within a tensor.

        The optional mean_range and std_range params allow for setting the range over which
        the distribution mean and std are sampled during each call. For special case where
        (x, x) is passed as the range we treat the value for that parameter as fixed.
        """
        if mean_range[1] < mean_range[0]:
            raise ValueError(
                f"Invalid mean_range parameter: upper bound must be >= lower bound: {mean_range}"
            )
        if std_range[1] < std_range[0]:
            raise ValueError(
                f"Invalid std_range parameter: upper bound must be >= lower bound: {std_range}"
            )

        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, tensor):
        mean = (
            self.mean_range[0]
            if self.mean_range[0] == self.mean_range[1]
            else torch.distributions.uniform.Uniform(*self.mean_range).sample()
        )
        std = (
            self.std_range[0]
            if self.std_range[0] == self.std_range[1]
            else torch.distributions.uniform.Uniform(*self.std_range).sample()
        )
        return tensor + torch.distributions.normal.Normal(loc=mean, scale=std).sample(
            tensor.shape
        ).to(DEVICE)
