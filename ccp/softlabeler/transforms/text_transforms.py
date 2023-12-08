from typing import Optional, Tuple

import numpy as np
import torch

from ccp.device_decision import DEVICE
from ccp.encoders.byte_pair_embed import BytePairEmbed


class ParagraphSwap(object):
    def __call__(self, tensor, index: Optional[int] = None):
        """
        Choose a random index along the length of a sequence of embedding vectors and
        swap the content above/below the index.

        Parameters:
        * tensor: tensor of shape <batch_size x 1 channel x sequence length x embedding dim>
        * index: optional index to use for the swap
        """
        if tensor.dim() != 4:
            raise RuntimeError(
                "ParagraphSwap expects 4-dimensional tensor: <batch_size x 1 channel x sequence length x dimensionality>"
            )
        sequence_length = tensor.shape[2]
        if index is None:
            index = np.random.randint(low=0, high=sequence_length)
        first_half = tensor[:, :, :index, :]
        second_half = tensor[:, :, index:, :]
        return torch.cat([second_half, first_half], dim=2)


class BPEmbRandomVectorSwap(object):
    def __init__(
        self,
        byte_pair_embedding: BytePairEmbed,
        replacement_range: Tuple[float, float] = (0.1, 0.25),
    ):
        """
        We replace embedding vectors (EVs) with a randomly chosen embedding vector
        from the full vocabulary defined by `byte_pair_embedding`.
        On each call, we choose a new number of EVs to replace, selecting uniformly at
        random from the range specified by `replacement_range`.
        """
        if replacement_range[1] < replacement_range[0]:
            raise ValueError(
                f"Invalid replacement_range parameter: upper bound must be >= lower bound: {replacement_range}"
            )
        if replacement_range[0] < 0:
            raise ValueError(
                f"Invalid replacement_range parameter: lower bound must be >= 0: {replacement_range}"
            )
        if replacement_range[1] > 1:
            raise ValueError(
                f"Invalid replacement_range parameter: upper bound must be <= 1: {replacement_range}"
            )
        self.replacement_range = replacement_range
        self.embedding = byte_pair_embedding

    def __call__(self, tensor):
        """
        Replaces particular EVs from `tensor` with entirely new EVs drawn at random
        from the space of known EVs.  This transform will tend to replace common EVs
        with less common EVs. The operation is in-place and will modify the
        source `tensor`.

        This transform operates on the entire batch simultaneously. For each token
        that is being swapped with a random EV from the vocabulary, we choose a token
        position `i`, and then replace the token at `i` with some new token value `t` for
        every sample in the batch. Although the tokens at position `i` differ by sample
        before this transform, the tokens at position `i` are identical between samples
        after this transform.

        :param tensor: shape <batch_size x 1 channel x sequence length x dimensionality>
        """
        if tensor.dim() != 4:
            raise RuntimeError(
                "BPEmbRandomVectorSwap expects 4-dimensional tensor: <batch_size x 1 channel x sequence length x dimensionality>"
            )

        num_words_in_sequence = tensor.shape[2]
        rate_of_evs_to_replace = (
            self.replacement_range[0]
            if self.replacement_range[0] == self.replacement_range[1]
            else torch.distributions.uniform.Uniform(*self.replacement_range).sample()
        )
        num_words_to_replace = int(num_words_in_sequence * rate_of_evs_to_replace)

        new_word_ids = torch.randint(
            low=0, high=self.embedding.vocab_size, size=(num_words_to_replace,)
        )
        new_word_embeddings = torch.Tensor(
            self.embedding.bpemb.vectors[new_word_ids]
        ).to(DEVICE)

        # guarantee that we don't select the same word repeatedly
        replacement_indices = torch.randperm(num_words_in_sequence)[
            :num_words_to_replace
        ]
        tensor[:, :, replacement_indices, :] = new_word_embeddings
        return tensor


class BPEmbVectorHide(object):
    def __init__(
        self,
        byte_pair_embedding: BytePairEmbed,
        replacement_range: Tuple[float, float] = (0.1, 0.25),
    ):
        """
        Randomly replace embedding vectors (EVs) with the learned padding vector
        used to pad short inputs. On each call, we choose a new number of EVs to replace,
        selecting uniformly at random from the range specified by `replacement_range`.

        :param byte_pair_embedding: the embedding space used for the EVs, from which
            we can retrieve the padding vector
        :param replacement_range: the range from which to sample a rate of EVs to replace
        """
        if replacement_range[1] < replacement_range[0]:
            raise ValueError(
                f"Invalid replacement_range parameter: upper bound must be >= lower bound: {replacement_range}"
            )
        if replacement_range[0] < 0:
            raise ValueError(
                f"Invalid replacement_range parameter: lower bound must be >= 0: {replacement_range}"
            )
        if replacement_range[1] > 1:
            raise ValueError(
                f"Invalid replacement_range parameter: upper bound must be <= 1: {replacement_range}"
            )
        self.replacement_range = replacement_range
        self.embedding = byte_pair_embedding

    def __call__(self, tensor):
        """
        Replaces an arbitrary number of EVs with the padding vector.
        This transform is in-place and will modify the source `tensor`.

        This transform operates on the entire batch simultaneously. For each token
        that is being swapped with a random EV from the vocabulary, we choose a token
        position `i`, and then replace the token at `i` with the learned padding vector
        used to pad short inputs. Although the tokens at position `i` differ by sample
        before this transform, the tokens at position `i` are identical between samples
        after this transform.

        Parameters:
        * tensor: tensor of shape <batch_size x 1 channel x sequence length x embedding dim>
        """
        if tensor.dim() != 4:
            raise RuntimeError(
                "BPEmbVectorHide expects 4-dimensional tensor: <batch_size x 1 channel x sequence length x dimensionality>"
            )
        num_words_in_sequence = tensor.shape[2]
        rate_of_evs_to_replace = (
            self.replacement_range[0]
            if self.replacement_range[0] == self.replacement_range[1]
            else torch.distributions.uniform.Uniform(*self.replacement_range).sample()
        )
        num_evs_to_replace = int(num_words_in_sequence * rate_of_evs_to_replace)

        # guarantee that we don't select the same word repeatedly
        replacement_indices = torch.randperm(num_words_in_sequence)[:num_evs_to_replace]
        # set value with broadcasting
        pad_embedding = torch.Tensor(
            self.embedding.bpemb.vectors[self.embedding.padding_id]
        ).to(DEVICE)
        tensor[:, :, replacement_indices, :] = pad_embedding
        return tensor
