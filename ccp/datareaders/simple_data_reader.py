import os
from typing import Any, Tuple

import numpy as np

from ccp.datareaders import DataReader
from ccp.typing import TargetLabel


class SimpleDataReader(DataReader):
    """
    Implementation of DataReader that handles a very simple data layout:
        One `.npz` file containing labelled data.
        One `.npz` file containing unlabelled data.
    Assumes that all data is already encoded/vectorized.
    """

    def __init__(self, labelled_npz_fpath: str, unlabelled_npz_fpath: str):
        """
        :param labelled_npz_fpath: Filepath to a `.npz` file containing labelled training examples.
            Expected to contain two numpy arrays of the same length, keyed `data` and `labels`.
        :param unlabelled_npz_fpath: Filepath to a `.npz` file containing unlabelled training examples.
            Expected to contain one numpy array, keyed 'data'.
        """
        if not os.path.isfile(labelled_npz_fpath):
            raise ValueError(f"Cannot find labelled filepath {labelled_npz_fpath}!")
        if not os.path.isfile(unlabelled_npz_fpath):
            raise ValueError(f"Cannot find unlabelled filepath {unlabelled_npz_fpath}!")

        with np.load(labelled_npz_fpath) as l_data:
            labelled_data = l_data["data"]
            labels = l_data["labels"]

        if labelled_data.shape[0] != labels.shape[0]:
            raise ValueError(
                "Provided label data has a different shape from provided labelled sample data!"
            )

        if DataReader.UNLABELLED_TARGET in labels:
            raise ValueError(
                f'Labels contain sentinel "unlabelled" target label {DataReader.UNLABELLED_TARGET}!'
            )

        with np.load(unlabelled_npz_fpath) as unl_data:
            unlabelled_data = unl_data["data"]

        self.data = np.concatenate((labelled_data, unlabelled_data))
        self.labels = np.concatenate(
            (labels, np.full(unlabelled_data.shape[0], DataReader.UNLABELLED_TARGET))
        )

        # Create class mapping:
        self.sorted_target_idxs_map = {
            target: np.sort(np.flatnonzero(np.asarray(self.labels == target)))
            for target in np.unique(self.labels)
        }

        super().__init__()

    def __getitem__(self, idx: int) -> Tuple[Any, TargetLabel]:
        return self.data[idx, :], self.labels[idx]

    @property
    def sorted_target_idxs(self):
        return self.sorted_target_idxs_map
