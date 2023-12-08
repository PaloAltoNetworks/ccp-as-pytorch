import math
import unittest

import pytest
import torch

from ccp.matrix_math import pairwise_angular_similarity, pairwise_cosine_similarity


class TestMatrixMath(unittest.TestCase):
    @pytest.mark.filterwarnings(
        "ignore: The use of `x.T`"
    )  # Expected warning for badly shaped tensor test
    def test_pairwise_cosine_sim(self):
        # Should raise error for > 2 dims, mismatched shapes:
        self.assertRaises(
            RuntimeError,
            lambda: pairwise_cosine_similarity(
                torch.rand((10, 5)), torch.rand((10, 4))
            ),  # MxN, MxD
        )
        self.assertRaises(
            RuntimeError,
            lambda: pairwise_cosine_similarity(
                torch.rand((10, 5, 4)), torch.rand((10, 5, 4))
            ),  # MxNxD, MxNxD
        )
        # Test that returns correctly for A != B, A == B:
        pairwise_cosine_similarity(torch.rand((10, 5)), torch.rand((10, 5)))
        A = torch.rand((10, 5))
        pairwise_cosine_similarity(A, A)

    def test_pairwise_cosine_similarity_example(self):
        A = torch.Tensor(
            [
                [1.0, 1.0],  # directionally x
                [0.5, 0.5],  # directionally x
                [-1.0, 0.0],  # directionally y
                [0, 0],  # has no direction
            ]
        )
        observed_out = pairwise_cosine_similarity(A, A)

        zero = 0
        one = 1
        sqrt2 = math.sqrt(2)
        # fmt: off
        expected_out = torch.tensor(
            [
                [one,      one,      -sqrt2/2, zero],  # noqa: E241,E226
                [one,      one,      -sqrt2/2, zero],  # noqa: E241,E226
                [-sqrt2/2, -sqrt2/2, one,      zero],  # noqa: E241,E226
                [zero,     zero,     zero,     zero],  # noqa: E241,E226
            ]
        )
        # fmt: on
        self.assertTrue(torch.isclose(expected_out, observed_out).all())

    def test_pairwise_angular_sim(self):
        # Failure modes already captured in `pairwise_cosine_similarity` above, which this calls...

        # Test that returns correctly for A != B, A == B:
        pairwise_angular_similarity(torch.rand((10, 5)), torch.rand((10, 5)))
        A = torch.rand((10, 5))
        pairwise_angular_similarity(A, A)

    def test_pairwise_angular_similarity_edge_case(self):
        """
        Verify that we do not get `nan` as the similarity, even if the inputs
        are almost identical.
        """
        # A is 2 rows that are almost but not quite identical
        A = torch.Tensor(
            [
                [-1.4901e-11, -1.4901e-11, -1.4901e-11],
                [-1.4901e-11, -1.4901e-11, -1.4902e-11],
            ]
        )

        # verify there are no NANs & we're seeing that the examples are functionally identical
        similarity_matrix = pairwise_angular_similarity(A, A)
        self.assertFalse(similarity_matrix.isnan().any())
        self.assertTrue(math.isclose(similarity_matrix.sum(), 4, rel_tol=1e-3))

    def test_pairwise_angular_sim_example(self):
        """
        Verify that similar positions yield high similarities.
        """
        reordered_z = torch.Tensor(
            [
                [1.0000, 1.0000],  # near (1,1)
                [-1.0000, -1.0000],  # near (-1, -1)
                [0.0000, 1.0000],  # near (0, 1)
                [0.9000, 1.1000],  # near (1,1)
                [-1.1000, -1.0100],  # near (-1, -1)
                [-0.0100, 1.0200],
            ]  # near (0, 1)
        )
        similarities = pairwise_angular_similarity(reordered_z, reordered_z)
        assert (similarities - torch.eye(6)).argmax(dim=0).tolist() == [
            3,
            4,
            5,
            0,
            1,
            2,
        ]
