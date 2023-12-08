import math

import torch
import torch.nn.functional as F


def pairwise_cosine_similarity(A, B, epsilon=1e-6):
    """
    Element pairwise similarity between two matrices A and B.
    Output range is [-1.0, 1.0] inclusive.
    """
    A_norm = F.normalize(A, p=2, dim=1)
    # No need to compute norm twice if pairwise comparing against self
    B_norm = A_norm if torch.equal(A, B) else F.normalize(B, p=2, dim=1)
    similarity = torch.mm(A_norm, B_norm.T)
    # without clamping, we can get values that look ok but are actually out-of-bounds
    return similarity.clamp(min=-1 + epsilon, max=1 - epsilon)


def pairwise_angular_similarity(A, B):
    """
    Element pairwise angular similarity between two matrices A and B.
    Output range is [0, 1] inclusive.
    """
    # arccos(x) has the domain [-1,1] and range [0,pi], so
    # when we divide by pi the range is [0,1]
    return 1 - (torch.arccos(pairwise_cosine_similarity(A, B)) / math.pi)
