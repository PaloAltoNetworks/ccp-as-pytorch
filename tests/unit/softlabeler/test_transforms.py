"""
Unit tests for the transforms module.
"""
import unittest

import torch

from ccp.softlabeler.transforms.generic_transforms import GaussianNoise
from ccp.softlabeler.transforms.text_transforms import (
    BPEmbRandomVectorSwap,
    BPEmbVectorHide,
    ParagraphSwap,
)


class TestGaussianNoise(unittest.TestCase):
    """
    Tests for the GaussianNoise transform.
    """

    def test_mean_unchanged(self):
        """
        Verify that we can execute the tranform without raising errors
        """
        gaussian_transform = GaussianNoise()
        gaussian_transform(torch.rand(30, 1, 512, 100))


class TestParagraphSwap(unittest.TestCase):
    """
    Tests for the ParagraphSwap transform.
    """

    def test_swap_example(self):
        paragraph_transform = ParagraphSwap()

        input_tensor = torch.Tensor(
            [
                [
                    [
                        [0, 10],
                        [1, 11],
                        [2, 12],
                        [3, 13],
                        [4, 14],
                        [5, 15],
                        [6, 16],
                        [7, 17],
                        [8, 18],
                        [9, 19],
                    ]
                ]
            ]
        )

        observed_out = paragraph_transform(input_tensor, index=2)
        expected_out = torch.Tensor(
            [
                [
                    [
                        [2, 12],
                        [3, 13],
                        [4, 14],
                        [5, 15],
                        [6, 16],
                        [7, 17],
                        [8, 18],
                        [9, 19],
                        [0, 10],
                        [1, 11],
                    ]
                ]
            ]
        )
        self.assertTrue(torch.equal(observed_out, expected_out))


class TestRandomBytePairEmbeddingVectorSwap(unittest.TestCase):
    """
    Tests for the BPEmbRandomVectorSwap transform.
    """

    def test_ev_swap(self):
        """
        Verify that we swap an appropriate number of indices.
        """
        from ccp.encoders.byte_pair_embed import BytePairEmbed

        BATCH_SIZE = 10
        VOCABULARY_SIZE = 50_000
        EMBEDDING_DIMENSIONALITY = 100
        FORCED_LENGTH = 512
        RATE_REPLACED = 0.1

        embed = BytePairEmbed(
            output_size=FORCED_LENGTH,
            vocab_size=VOCABULARY_SIZE,
            embedding_dimensionality=EMBEDDING_DIMENSIONALITY,
        )
        transform = BPEmbRandomVectorSwap(
            embed, replacement_range=(RATE_REPLACED, RATE_REPLACED)
        )

        t = torch.rand(BATCH_SIZE, 1, FORCED_LENGTH, EMBEDDING_DIMENSIONALITY)
        original_t = t.clone()
        observed_t = transform(t)
        observed_replaced_indices = (
            (original_t != observed_t).sum(dim=(0, 1, 3))
            == BATCH_SIZE * EMBEDDING_DIMENSIONALITY
        ).nonzero()
        observed_num_replaced = len(observed_replaced_indices)

        expect_num_replaced = int(RATE_REPLACED * FORCED_LENGTH)
        self.assertEqual(observed_num_replaced, expect_num_replaced)


class TestBPEmbVectorHide(unittest.TestCase):
    """
    Tests for the BPEmbVectorHide transform.
    """

    def test_ev_hide(self):
        """
        Verify that we hide an appropriate number of EVs with padding vector.
        """
        from ccp.encoders.byte_pair_embed import BytePairEmbed

        BATCH_SIZE = 10
        VOCABULARY_SIZE = 50_000
        EMBEDDING_DIMENSIONALITY = 100
        FORCED_LENGTH = 512
        RATE_REPLACED = 0.9

        embed = BytePairEmbed(
            output_size=FORCED_LENGTH,
            vocab_size=VOCABULARY_SIZE,
            embedding_dimensionality=EMBEDDING_DIMENSIONALITY,
        )
        transform = BPEmbVectorHide(
            embed, replacement_range=(RATE_REPLACED, RATE_REPLACED)
        )

        t = torch.randn(BATCH_SIZE, 1, FORCED_LENGTH, EMBEDDING_DIMENSIONALITY)
        original_t = t.clone()
        observed_t = transform(t)
        observed_replaced_indices = (
            (original_t != observed_t).sum(dim=(0, 1, 3))
            == BATCH_SIZE * EMBEDDING_DIMENSIONALITY
        ).nonzero()
        observed_num_replaced = len(observed_replaced_indices)

        expect_num_replaced = int(RATE_REPLACED * FORCED_LENGTH)
        self.assertEqual(observed_num_replaced, expect_num_replaced)
