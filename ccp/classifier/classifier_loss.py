import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    """
    Custom soft cross entropy loss for classification tasks using soft credibility labels,
    as defined in CCP.
    This is a cross-entropy loss similar to SEAL loss using q-values rather than true
    label vectors. It averages across all batch elements and classes, weighting performance
    by the q-vector likelihood of each cell.
    """

    def __init__(self, epsilon: float = 1e-8):
        super(SoftCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, gij, qij):
        if gij.dim() != 2 or qij.dim() != 2:
            raise ValueError("Inputs to SoftCrossEntropyLoss must be 2D tensors!")
        clamped_qij = torch.clamp(qij, min=0.0, max=1.0)
        return -torch.mean(
            clamped_qij * torch.log(F.softmax(gij, dim=1) + self.epsilon), dim=(0, 1)
        )
