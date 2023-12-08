from itertools import repeat
from typing import List

from bpemb import BPEmb
from torch import nn, tensor

from ccp.device_decision import DEVICE


class BytePairEmbed(object):
    """
    Embed text using byte-pair embeddings. Crop and/or pad embedding
    vectors to be of size `output_size`.
    """

    def __init__(
        self,
        output_size: int,
        lang="en",
        vocab_size=50_000,
        embedding_dimensionality=100,
    ):
        """
        Embed text using byte-pair embeddings.

        :param output_size: the sequence length to which every example
            will be padded; examples longer than this length will be truncated
        :param lang: 2-letter country code for language of text; this is a
            pass-through for a BPEmb parameter
        :param vocab_size: vocabulary size for byte pair embeddings; this is a
             pass-through for a BPEmb parameter
        :param embedding_dimensionality: embedding size for byte pair embeddings;
            this is a pass-through for a BPEmb parameter
        """
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding_dimensionality = embedding_dimensionality

        self.bpemb = BPEmb(
            lang=lang, vs=vocab_size, dim=embedding_dimensionality, add_pad_emb=True
        )
        self.padding_id = (
            vocab_size  # always added as last element when `add_pad_emb=True`
        )

        # we do not freeze the embedding layer because the CCP paper states
        # "[we] look up an embedding vector whose *initial values* are set to the
        # pretrained BPEmb values" (emphasis added)
        self.embedding_layer = nn.Embedding.from_pretrained(
            tensor(self.bpemb.vectors), padding_idx=self.padding_id, freeze=False
        )

    def __call__(self, batch_text: List[str]):
        """
        Given an iterable of BATCH_SIZE text strings, returns the padded or
        truncated embedded versions of those text strings.

        Input:
            BATCH_SIZE x 1, each cell is a string
        Output:
            BATCH_SIZE x self.output_size x self.embedding_dimensionality
        """
        batch_ids = []
        for i, sample in enumerate(batch_text):
            ids = self.bpemb.encode_ids(sample)
            ids = ids[: self.output_size]

            amount_of_padding = self.output_size - len(ids)
            ids = ids + list(repeat(self.padding_id, amount_of_padding))
            batch_ids.append(ids)

        return self.embedding_layer(tensor(batch_ids).to(DEVICE))
