import logging
from collections import deque
from typing import Dict, Optional

import torch
from torch.utils.data import Sampler

from ccp.typing import TargetLabel, TargetLabelIdxs

LOGGER = logging.getLogger(__name__)


class CCPBatchSampler(Sampler):
    """
    A custom batch sampler for loading CCP sample batches from a CCPDataset instance.

    The sampler will return batches of size `batch_size` or potentially less than `batch_size` (for the last batch).
    """

    def __init__(
        self,
        batch_size: int,
        dataset_target_idxs: TargetLabelIdxs,
        target_sample_rates: Dict[TargetLabel, int] = {},
        epoch_target_cond: Optional[TargetLabel] = None,
        random_seed: int = 42,
    ):
        """
        By default, CCPBatchSampler will keep offering batches until every single row has been sampled at least once.
        In order to focus on a single target label, you can set it as the `epoch_target_cond` and CCPBatchSampler will
         instead keep offering batches until all rows for that target have been sampled at least once.
        :param batch_size: The size of the batches to sample.  Note that batches are only guaranteed to be at most this
            size (in particular the last batch is unlikely to be of size `batch_size`). This matches the paper; it may
            be better to always sample complete batches.
        :param dataset_target_idxs: A mapping of unique target labels in the dataset to sample to the indices of
         samples of that target.  Must include all target labels and indices in the dataset - used to construct batches.
        :param target_sample_rates:  A dictionary mapping target labels to sample rates.
          Any non-existent keys are assumed to be sampled at a rate of `1`, ie roughly balanced in a batch.
          A sample rate of 10 means take 10 samples of that target for each single pass over the targets when
          constructing a batch.  Default is fully balanced. Set to 0 to exclude a target (e.g., CCPDataset.UNLABELLED_TARGET).
        :param random_seed: Seed for numpy random - set for reproducible randomness.
        :param epoch_target_cond: The epoch condition target - by default (None) one epoch of batches is completed when
            all samples in the dataset have been included in a batch.  However, if a specific target is provided, then
            batches will only be generated until all samples *of that target label* have appeared once.
        """
        # Construct a full class_sample_rates dictionary accounting for all classes, defaulting 1s:
        self.target_sample_rates = {
            target: target_sample_rates[target] if target in target_sample_rates else 1
            for target in dataset_target_idxs.keys()
        }

        if batch_size < sum(self.target_sample_rates.values()):
            raise ValueError(
                f"Batch size of {batch_size} is smaller than number of examples need to achieve your requested per-batch sample ratios (class : desired # examples of class in each batch is {self.target_sample_rates}) - can't build batches!"
            )

        if epoch_target_cond and epoch_target_cond not in dataset_target_idxs:
            raise ValueError(
                f"Unknown epoch class {epoch_target_cond}! Target must occur in dataset_target_idxs."
            )
        self.epoch_target_cond = epoch_target_cond

        num_elements_in_an_epoch = sum(
            [
                self.target_sample_rates[k] * len(v)
                for k, v in dataset_target_idxs.items()
            ]
        )
        if batch_size > num_elements_in_an_epoch:
            raise ValueError(
                f"Max batch size is {num_elements_in_an_epoch} (there are {num_elements_in_an_epoch} samples)"
            )

        self.batch_size = batch_size
        self.dataset_target_idxs = dataset_target_idxs
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)  # Set random seed for repeatability

    def __iter__(self):
        self._generate_batches()
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    @property
    def batches(self):
        if not hasattr(self, "_batches"):
            raise ValueError(
                "Call `iter()` on the sampler object itself to build batches!"
            )
        return self._batches

    def _generate_batches(self):
        """
        Generate an "epoch" batches of samples randomly.
        When building batches, we are guaranteed to pass over all of the targets at least once because we check on
        __init__ that batch_size is large enough to support all classes at sample rate.
        Thus each batch is guaranteed to have at least sample_rate[target] examples from each target.
        Further, each batch is guaranteed to be of size `batch_size` except for the last batch, which will be of
        size sum(sample_rate[target]*target for targets).
        If epoch_target_cond is not None, then most samples of target `epoch_target_cond` will appear exactly once (but
            some may be sampled a second time in the last batch).
        Else this limit applies to the single largest target by sample rate (ie
            argmax(len(class_idxs[target])/sample_rate[target]).
        This function will set the self._batches property on the class to the generated batches.
        """

        def epoch_cond(fully_sampled_dict: Dict[TargetLabel, bool], target_key):
            """
            Terminating condition for batch generation.
            If target_key is provided, checks that this target is "fully sampled" according to `fully_sampled_dict`
            (ie value is True) - else, checks that all targets are  "fully sampled" according to `fully_sampled_dict`.
            :param fully_sampled_dict: Boolean dictionary mapping target keys to "fully sampled" status (True/False).
            :param target_key: Optionally a target key to uniquely use as an epoch sampling condition.
            :return True if epoch condition is reached, False otherwise.
            """
            return (
                fully_sampled_dict[target_key]
                if target_key is not None
                else all(fully_sampled_dict.values())
            )

        # First, randomize the order of each class - use deque instead of generator because we need to know when empty:
        target_random_idxs = {
            target: deque((idxs[r_ind] for r_ind in torch.randperm(len(idxs))))
            for target, idxs in self.dataset_target_idxs.items()
        }
        is_target_fully_sampled = {
            target: (self.target_sample_rates[target] <= 0)
            for target in self.dataset_target_idxs.keys()
        }
        targets = list(target_random_idxs.keys())

        # Iterate through targets repeatedly until we reach a batch size, or until epoch_cond():
        batches = []
        cur_batch = []
        while not epoch_cond(is_target_fully_sampled, self.epoch_target_cond):
            for t_idx in torch.randperm(len(targets)):
                target = targets[t_idx]
                # Sample "class_sample_rates" times, defaulting at 1:
                for _ in range(self.target_sample_rates[target]):
                    idx_dq = target_random_idxs[target]
                    if len(cur_batch) == self.batch_size:
                        break

                    idx = idx_dq.popleft()
                    cur_batch.append(idx)

                    if len(idx_dq) == 0:
                        # Set class_fully_sampled, and build a new deque of idxs:
                        is_target_fully_sampled[target] = True
                        # Technically this allows the sample multiple times in a batch, but this is hard to avoid
                        target_idxs = self.dataset_target_idxs[target]
                        target_random_idxs[target] = deque(
                            (
                                target_idxs[r_ind]
                                for r_ind in torch.randperm(len(target_idxs))
                            )
                        )

                if len(cur_batch) == self.batch_size:
                    if cur_batch:
                        batches.append(cur_batch)
                    cur_batch = []
                    break
        self._batches = batches
