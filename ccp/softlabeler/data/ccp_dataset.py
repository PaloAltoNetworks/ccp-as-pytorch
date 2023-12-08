from typing import Any, Tuple

import torch
from torch.utils.data import Dataset

from ccp.classifier.q_label_dataset import QLabelDataset
from ccp.datareaders import DataReader
from ccp.softlabeler.data.q_vector_handler import QVectorHandler
from ccp.typing import CCPMetadata, TargetLabel


class CCPDataset(Dataset):
    """
    Class defining a dataset for CCP training.  Users of the class provide a DataReader implementation that handles all
    sample retrieval and target/sample encoding (target -> integer label, sample to np array or tensor), and this class
    leverages that functionality to provide CCP with the expected training data.

    This class additionally owns a QVectorHandler instance - this object handles all Q-Vector operations.
    This class exposes a series of pass-through methods that interact with the internals of the QVectorHandler, and are
    used by the ContrastiveCredibilityLabeller to propagate q labels.
    """

    def __init__(self, data_reader: DataReader):
        """
        :param data_reader: A concrete implementation of a DataReader - used by the dataset to retrieve samples
            and a target mapping.
        """
        data_reader.validate()
        self.data_reader = data_reader

    def __len__(self) -> int:
        """
        Return total number of rows (labeled + unlabeled).
        """
        return self.data_reader.n_labelled + self.data_reader.n_unlabelled

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, TargetLabel, int]:
        sample_X: Any
        target_y: TargetLabel
        sample_X, target_y = self.data_reader.__getitem__(idx)
        q_vec = self.q_handler.get_q_vec(idx)
        return sample_X, q_vec, target_y, idx

    def to_classification_dataset(self) -> QLabelDataset:
        """
        Helper function that will turn a CCPDataset into a QLabelDataset.
        This mainly involves freezing the q-vectors, and then building a QLabelDataset from
        these frozen soft labels and a reference to the DataReader implementation.
        :return: A QLabelDataset instance that can be used for training a classifier.
        """
        return QLabelDataset(
            data_reader=self.data_reader, q_vecs=self.q_handler.q_vecs.detach().clone()
        )

    @property
    def q_handler(self) -> QVectorHandler:
        if not hasattr(self, "_q_handler"):
            self._q_handler = QVectorHandler(
                n_samples=len(self),
                sorted_target_idxs=self.data_reader.sorted_target_idxs,
                unlabelled_target=DataReader.UNLABELLED_TARGET,
            )
        return self._q_handler

    # PASSTHROUGH FUNCTIONS INTERACTING DIRECTLY WITH QVECTORHANDLER:

    def propagate_batch_q_vecs(
        self,
        z: torch.Tensor,
        q: torch.Tensor,
        targets: torch.Tensor,
        idxs: torch.Tensor,
    ):
        """
        Pass-through to QVectorHandler batch propagation function.
        :param z: The projected, transformed samples for the batch.  Expected as a stack of:
            [z(t1(s1)), z(t1(s2)), ..., z(t2(s1)), z(t2(s2)), ...]
        :param q: The q-vectors for samples in the batch. Stacked to correspond to elements in z.
        :param targets:  The targets corresponding to each sample in batch.  Stacked to correspond to elements in z.
        :param idxs:  The idx for each sample into the *overall* dataset.  Note this is an *unduplicated* tensor,
            so it will be half as long as the others.  E.g.: [idx(s1), idx(s2), ..., idx(sn)]
        """
        self.q_handler.propagate_batch_q_vecs(z=z, q=q, targets=targets, idxs=idxs)

    def propagate_q_vecs(self, metadata: CCPMetadata):
        """
        Pass-through to QVectorHandler propagation function.
        :param metadata: The metadata to use for current q-vector propagation.
        :return: Metadata for next CCP propagation run.
        """
        return self.q_handler.propagate_q_vecs(metadata=metadata)

    def write_q_vecs(self, output_directory: str, output_fname: str):
        """
        Pass through to QVectorHandler persistence function for writing Q-vector results.
        """
        self.q_handler.write_q_vecs(output_dir=output_directory, filename=output_fname)
