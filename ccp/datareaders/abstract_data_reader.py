from abc import ABC, abstractmethod
from typing import Any, Tuple

from ccp.typing import TargetLabel, TargetLabelIdxs


class DataReader(ABC):
    """
    Abstract interface for defining data sample loading operations.  Users of CCP can define subclasses implementing
    the methods of this class in whatever way best fits access patterns for individual samples of their dataset.
    For example:
        1. Reading and storing all data in memory to return as simple lookups.
        2. Reading individual files from disk per sample (lookup to filepath).

    The intention of this class is to provide a single implementation for data access operations - implementations
    can then be leveraged both by the core CCP loop (in a CCPDataset) and during subsequent classification tasks
    with learned soft labels (in a QLabelDataset).
    """

    UNLABELLED_TARGET: TargetLabel = -1  # type: ignore

    @property
    @abstractmethod
    def sorted_target_idxs(self) -> TargetLabelIdxs:
        """
        Return a mapping of TargetLabel (INTEGER) targets to corresponding indices into the Dataset instance.
        Note that this means the label preprocessing (categorical class -> numerical target) should happen upstream of
        this function.
        This mapping is expected to have the properties outlined in self._validate_target_idxs(), which you can verify
        by running that method.

        This mapping serves as the source for idxs during batch sampling.
        The sentinel key CCP.UNLABELLED_TARGET is used to map indices that are unlabelled.
        :return: A dictionary mapping of targets to indices into the Dataset for samples of that target type,
            adhering to the 3 properties above.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, TargetLabel]:
        """
        Return the encoded data example at idx and the corresponding TargetLabel.
        "Encoded" data should be a numpy array or tensor, ready to pass directly in batches through transformations
         and into the encoder network.
        This function should be implemented by subclasses, and is called in the base __getitem__ implementation
        to fetch data examples.
        """
        pass

    @property
    def n_labelled(self) -> int:
        if not hasattr(self, "_n_labelled"):
            self._n_labelled = sum(
                map(
                    len,
                    (
                        idxs
                        for target, idxs in self.sorted_target_idxs.items()
                        if target != DataReader.UNLABELLED_TARGET
                    ),
                )
            )
        return self._n_labelled

    @property
    def n_unlabelled(self) -> int:
        if not hasattr(self, "_n_unlabelled"):
            self._n_unlabelled = len(
                self.sorted_target_idxs[DataReader.UNLABELLED_TARGET]
            )
        return self._n_unlabelled

    @property
    def n_distinct_labels(self) -> int:
        if not hasattr(self, "_n_distinct_labels"):
            self._n_distinct_labels = len(self.sorted_target_idxs) - 1
        return self._n_distinct_labels

    def validate(self) -> None:
        """
        Validates the implementation of the DataReader class. If the implementation
        is valid, this method is a no-op. If it is invalid, it will raise an exception.
        """
        self._validate_target_idxs(self.sorted_target_idxs)

    @staticmethod
    def _validate_target_idxs(sorted_target_idxs):
        """
        Simple validation on sorted_target_idxs mapping.
        1. Each idx between 0-(max(idx)-1) is mapped to a target class (concretely, idxs are expected to be an int range
            from 0 - (# of samples - 1)).
        2. Each idx mapping must be sorted in ascending order (lowest idx to highest for target).
        3. Each idx mapping should contain no duplicates.
        4. No idx can be mapped to multiple targets.
        5. The sentinel key DataReader.UNLABELLED_TARGET must exist (can map to an empty array).
        :return: Raises a ValueError if any of these properties is violated.
        """
        # Property (5)
        if DataReader.UNLABELLED_TARGET not in sorted_target_idxs:
            raise ValueError("Property target_idxs does not map UNLABELLED_TARGET!")

        seen_idx = set()
        for target, idxs in sorted_target_idxs.items():
            if any((idx < 0 for idx in idxs)):
                raise ValueError(f"Target {target} contains a non-positive idx!")

            target_idxs_set = set(idxs)
            # Property (3)
            if len(target_idxs_set) != len(idxs):
                raise ValueError(
                    f"Target {target} in property target_idxs contains duplicates in idx mapping!"
                )

            # Property (4)
            if seen_idx & target_idxs_set:
                raise ValueError(
                    f"Target {target} in property target_idxs contains already mapped idxs: {seen_idx & target_idxs_set}!"
                )
            seen_idx |= target_idxs_set

            # Property (2)
            if not all(idxs[i] < idxs[i + 1] for i in range(len(idxs) - 1)):
                raise ValueError(
                    f"Target {target} in property target_idxs contains unsorted idx mapping!"
                )

        # Property (1)
        if (
            0 not in seen_idx or max(seen_idx) != len(seen_idx) - 1
        ):  # uniqueness constraint + > 0 constraint
            raise ValueError(
                "DataReader idxs should correspond to elements in range(num_samples)!"
            )
