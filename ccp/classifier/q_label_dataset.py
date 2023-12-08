from typing import Any, Tuple

import torch
from torch.utils.data import Dataset

from ccp.datareaders import DataReader


class QLabelDataset(Dataset):
    """
    Dataset class implementation intended for use training a SoftLabel classifier based on CCP-produced Q-labels.
    Note that this dataset expects to operate over some subset of the full q-vector/sample space passed in on
    initialization.
    """

    def __init__(self, data_reader: DataReader, q_vecs: torch.Tensor):
        """
        :param data_reader: A concrete implementation of a DataReader - used by the dataset to retrieve samples
            and a target mapping.
        :param q_vecs: A copy of the soft labels (q-vectors) corresponding to samples in the provided DataReader.
            Expected as a single tensor with dimensions (n_samples x n_classes).
        """
        data_reader.validate()
        self.data_reader = data_reader
        # Figure out which q_vectors are worth keeping, and construct a new smaller q_vec tensor:
        # For now we just drop zeroed labels - we may eventually want to allow a strength param for
        # additional filtering.
        non_zero_mask = torch.any(q_vecs, dim=1)

        # We need to maintain a map from this dataset's idx space [0, ... len(self)-1] to the DataReader idx space, and
        # ensure that the compressed q-vector idx space can link to it correctly (the new q-vector space is identical to
        # the overall dataset's idx space, since we compress out zero entries).
        self.idx_mapping = {
            q_idx: sample_idx
            for q_idx, sample_idx in enumerate(
                non_zero_mask.nonzero()
            )  # Retrieve pos of True elems
        }
        self.q_vecs = q_vecs[non_zero_mask]

    def __len__(self) -> int:
        """
        Return total number of rows.
        """
        return len(self.idx_mapping)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        sample: Any
        sample_idx = self.idx_mapping[idx]
        sample, _ = self.data_reader.__getitem__(sample_idx)
        q_vec = self.q_vecs[idx]
        return sample, q_vec
