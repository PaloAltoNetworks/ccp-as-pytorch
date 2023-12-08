import unittest

import torch

from ccp.classifier.classifier_loss import SoftCrossEntropyLoss


class TestSoftCrossEntropyLoss(unittest.TestCase):
    def test_soft_sup_con(self):
        criterion = SoftCrossEntropyLoss()
        # Test failure modes:
        self.assertRaises(
            ValueError,
            lambda: criterion.forward(
                torch.rand((10, 5, 4)), torch.rand((10, 5, 4))
            ),  # MxNxD, MxNxD
        )
        criterion.forward(torch.rand(10, 5), torch.rand(10, 5))
