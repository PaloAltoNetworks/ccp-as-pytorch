import unittest

import torch

from ccp.encoders.byte_pair_embed import BytePairEmbed
from ccp.encoders.ccp_text_encoder import TextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_encoder_shape(self):
        """
        Verify the encoder runs forward without raising exceptions, and
        that it produces the right shape.

        NOTE: this test issues a network call to get embeddings if they aren't cached
        """
        max_sequence_length = 512
        embed_dimensionality = 100
        conv1_output_size = 16
        conv2_output_size = 16

        embedder = BytePairEmbed(
            output_size=max_sequence_length,
            embedding_dimensionality=embed_dimensionality,
        )
        encoder = TextEncoder(
            dim_in=(max_sequence_length, embed_dimensionality),
            conv1b_num_filters=conv1_output_size,
            conv2b_num_filters=conv2_output_size,
        )

        string_input = [
            "She sells seashells on the sea shore",
            "Fischers Fritz fischt frische Fische, frische Fische fischt Fischers Fritz",
        ]
        # add an extra dimension representing that this input has a single channel
        embedded_input = embedder(string_input)[:, None, :, :]
        assert embedded_input.shape == torch.Size(
            [len(string_input), 1, max_sequence_length, embed_dimensionality]
        )
        b = encoder(embedded_input)
        assert b.shape == torch.Size(
            [len(string_input), conv1_output_size + conv2_output_size]
        )
