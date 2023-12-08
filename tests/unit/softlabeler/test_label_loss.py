import unittest

import torch

from ccp.softlabeler.label_loss import SoftSupervisedContrastiveLoss


class TestSoftSupervisedContrastiveLoss(unittest.TestCase):
    def test_soft_sup_con(self):
        criterion = SoftSupervisedContrastiveLoss()
        # Test failure modes:
        self.assertRaises(
            ValueError,
            lambda: criterion.forward(
                torch.rand((10, 5, 4)), torch.rand((10, 5, 4))
            ),  # MxNxD, MxNxD
        )
        criterion.forward(torch.rand(10, 5), torch.rand(10, 3))
