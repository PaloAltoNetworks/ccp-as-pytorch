from typing import Callable

import torch
import torch.nn as nn

from ccp.device_decision import DEVICE
from ccp.matrix_math import pairwise_angular_similarity


class SoftSupervisedContrastiveLoss(nn.Module):
    """
    Custom soft supervised contrastive loss implementation, as defined in CCP.
    This is a fully supervised loss function that generalizes SimCLR and SupCon losses.

    See: Equation 2 for L_SSC
    """

    def __init__(
        self,
        similarity_function: Callable = pairwise_angular_similarity,
        temperature: float = 0.1,
        epsilon: float = 1e-6,
    ):
        """
        Creates an instance of the soft supervised contrastive loss defined in CCP.

        :param similarity_function: a pairwise similarity function that calculates
            the similarity between z vectors representing information relevant to the
            label space that are produced by the CCP projection head; the choice of
            similarity function directly affects the accuracy of pseudo-labels; this
            method must take in two matrices and produce their elementwise
            pairwise similarity
        :param temperature: a scaling factor for pairwise similarities
        :param epsilon: a small zero-like value, which accounts for translating analytic
            linear algebra into computational floats
        """
        super(SoftSupervisedContrastiveLoss, self).__init__()
        self.pairwise_similarity_function = similarity_function
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, zij, qij):
        if zij.dim() != 2 or qij.dim() != 2:
            raise ValueError(
                "Inputs to SoftSupervisedContrastiveLoss must be 2D tensors!"
            )
        clamped_qij = torch.clamp(qij, min=0.0, max=1.0)
        num_samples = clamped_qij.shape[0]

        # Pairwise matching matrix M, with self-pairs zeroed:
        M = torch.mm(clamped_qij, clamped_qij.T) * (
            1 - torch.eye(num_samples, device=DEVICE)
        )
        # Compute pairwise similarities scale by temperature and take the exponential, zero self-pairs:
        A = self.pairwise_similarity_function(zij, zij)
        A = torch.exp(A / self.temperature) * (
            1 - torch.eye(num_samples, device=DEVICE)
        )
        # Compute strength matrix, with self-pairs zeroed:
        S = torch.tile(torch.amax(clamped_qij, dim=1, keepdim=True), (zij.shape[0],))
        # Scale and normalize similarities A using q confidence (S):
        A_inner = A / (torch.sum(A * S, dim=1, keepdim=True) + self.epsilon)
        A = torch.log(A_inner.clamp(min=1e-6))  # clamp to domain of log function

        # Compute batch loss scaled by confidence (just one section from tiled strength matrix):
        return torch.mean(
            S[:, :1] * (-torch.sum(A * M, dim=1) / (torch.sum(M, dim=1) + self.epsilon))
        )
