import logging
from typing import Callable, Dict, List, Optional, Sized, Tuple

import numpy as np
from torchtext.datasets import AG_NEWS  # DataPipe is beta status as of November 2022
from tqdm import tqdm

from ccp.datareaders import DataReader
from ccp.typing import TargetLabel, TargetLabelIdxs

LOGGER = logging.getLogger(__name__)


class AgNewsDataReader(DataReader):
    def __init__(
        self,
        split: str,
        rate_to_keep: float = 1.0,
        transform: Optional[Callable] = None,
        random_seed: int = 42,
        sample_indices_to_use: Optional[List[int]] = None,
    ):
        """
        Builds DataReader from AGNews, in which we can "forget" some
        number of labels (1-`rate_to_keep`).

        AGNews is a small dataset; there are 120k examples in train, 7.6k examples in
        test. Train idx_to_filename is balanced among 4 classes.

        The classes in AGNews are: World, Sports, Business, Sci/Tech (in no particular
        order; this datareader varies the indices so that the first encountered item
        is always class 0).

        Parameters:
        `rate_to_keep`: how many labels to retain as hard labels
            (rate is applied per class, so two classes with the same
            number of examples in input will have the same number
            of examples after forcing some labels to be forgotten)
        `split`: can be "train" or "test" (passed to the torchtext dataset)
        `transform`: a Callable that executes on each example to convert it
            to a featurized Tensor; if None, each example is the raw string
        `random_seed`: seed for selecting kept vs. forced-unlabeled
            examples
        `sample_indices_to_use`: optional; if None, include all the indices
            part of our dataset; if a list of integers, only consider the values
            at those indices as part of our dataset (this param is useful for reducing
            the dataset size during testing)
        """
        if rate_to_keep <= 0 or rate_to_keep > 1:
            raise ValueError(
                f"`rate_to_keep` must be between 0 and 1, not {rate_to_keep}"
            )
        self.transform = transform

        self.data: List[str] = []
        self.labels: List[TargetLabel] = []  # type: ignore
        self.map_official_label_to_dest_label: Dict[int, TargetLabel] = {}  # type: ignore
        self.__target_idxs_map: TargetLabelIdxs = {}

        for idx, (source_label, line) in enumerate(tqdm(AG_NEWS(split=split))):
            if sample_indices_to_use and idx not in sample_indices_to_use:
                continue
            self.data.append(line)

            if source_label not in self.map_official_label_to_dest_label:
                dest_label = len(self.map_official_label_to_dest_label)
                self.map_official_label_to_dest_label[source_label] = dest_label
                self.__target_idxs_map[dest_label] = np.array([]).astype(int)
            dest_label: TargetLabel = self.map_official_label_to_dest_label[source_label]  # type: ignore
            self.labels.append(dest_label)

            # Note that these are guaranteed to be in sorted order!
            self.__target_idxs_map[dest_label] = np.append(
                self.__target_idxs_map[dest_label], len(self.labels) - 1
            )

        LOGGER.info(
            f"Target: num examples after load: "
            f"{ {target: len(idxs) for target, idxs in self.__target_idxs_map.items()} }"
        )
        if rate_to_keep < 1:
            self.__forget_labels(rate_to_keep, random_seed)
        else:
            self.__target_idxs_map[self.UNLABELLED_TARGET] = np.array([])

        super().__init__()

    def __forget_labels(self, rate_to_keep: float, random_seed: int) -> None:
        """
        Update internal state so that labels are forgotten, that is,
        the labels are replaced with DataReader.UNLABELLED_TARGET.
        """
        random_state = np.random.RandomState(seed=random_seed)
        self.labels = np.array(self.labels)  # type: ignore  # recast of type
        self.__target_idxs_map[DataReader.UNLABELLED_TARGET] = np.array([]).astype(int)

        all_targets = set(self.__target_idxs_map.keys()) - set(
            [DataReader.UNLABELLED_TARGET]
        )
        for our_target in sorted(
            all_targets
        ):  # defined order for random reproducibility
            indices_for_target = self.__target_idxs_map[our_target]
            num_kept_hard_labels = int(rate_to_keep * len(indices_for_target))
            if num_kept_hard_labels == 0:
                raise ValueError(
                    f"No examples of CCP label {our_target} (orig count {len(indices_for_target)}) "
                    f"persist under rate_to_keep of {rate_to_keep}!"
                )

            indices_for_target = random_state.permutation(indices_for_target)
            indices_of_hard_labels = indices_for_target[:num_kept_hard_labels].astype(
                int
            )
            indices_of_forgotten_labels = indices_for_target[
                num_kept_hard_labels:
            ].astype(int)

            self.labels[indices_of_forgotten_labels] = DataReader.UNLABELLED_TARGET
            self.__target_idxs_map[our_target] = indices_of_hard_labels
            self.__target_idxs_map[DataReader.UNLABELLED_TARGET] = np.hstack(
                [
                    self.__target_idxs_map[DataReader.UNLABELLED_TARGET],
                    indices_of_forgotten_labels,
                ]
            ).astype(int)

            LOGGER.info(
                f"Effect of forgetting on class {our_target}: from "
                f"{len(indices_for_target)} to {len(indices_of_hard_labels)} examples"
            )
        LOGGER.info(
            f"The unlabeled class {DataReader.UNLABELLED_TARGET} contains "
            f"{len(self.__target_idxs_map[DataReader.UNLABELLED_TARGET])} examples"
        )

        for target in self.__target_idxs_map.keys():
            self.__target_idxs_map[target] = np.sort(self.__target_idxs_map[target])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Sized, TargetLabel]:
        """
        Returns the (sample_X, label_y) at idx, where sample_X has been
        transformed according to self.transform if self.transform was set.

        The `sample_X` result is guaranteed to have a first dimension/len of 1,
        reflecting that this is a single sample. The shape of any other dimensions
        will vary based on self.transform.
        """
        sample_X = self.data[idx]
        label_y = self.labels[idx]

        if self.transform:
            return self.transform([sample_X]), label_y
        else:
            return [sample_X], label_y

    @property
    def sorted_target_idxs(self) -> TargetLabelIdxs:
        return self.__target_idxs_map
