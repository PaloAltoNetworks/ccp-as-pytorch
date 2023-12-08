import logging
import os
from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
from scipy.stats import entropy  # type: ignore

from ccp.device_decision import DEVICE
from ccp.matrix_math import pairwise_angular_similarity
from ccp.typing import CCPMetadata, IdxMask, TargetLabel, TargetLabelIdxs

LOGGER = logging.getLogger(__name__)


class QVectorHandler(object):
    """
    Logical handler for operations related to Q-vectors.
    This has a tight coupling with a CCPDataset instance, since these Q-Vectors correspond to data examples.

    The internal details of this class are unlikely to be useful to users - the only exception is `save_q_vecs`, which
    will write current q-vector state to disk at the provided path.
    """

    KL_DIVERGENCE_MAX_SCALING_FACTOR = 0.1
    P_SHIFT = 0.01

    def __init__(
        self,
        n_samples: int,
        sorted_target_idxs: TargetLabelIdxs,
        unlabelled_target: TargetLabel,
    ):

        self.n_samples = n_samples

        self._sorted_target_idxs = sorted_target_idxs
        self.unlabelled_target = unlabelled_target

        self.labelled_target_idxs = {
            target: idxs
            for target, idxs in self._sorted_target_idxs.items()
            if target != unlabelled_target
        }

        self.n_targets = len(self.labelled_target_idxs)
        self.n_unlabelled_targets = len(
            self._sorted_target_idxs[self.unlabelled_target]
        )

        # Additionally extract all "labelled" idxs into a set to track which q-vectors should be fixed:
        self.labelled_idxs = set(
            chain.from_iterable(self.labelled_target_idxs.values())
        )

        LOGGER.info(f"Considering {self.n_unlabelled_targets} unlabelled samples.")

        # Initialize q-vectors:
        self.q_vecs = torch.zeros((n_samples, self.n_targets), device=DEVICE)

        for target, instance_idxs in self.labelled_target_idxs.items():
            labelled_q_vec = torch.zeros(self.n_targets, device=DEVICE)
            labelled_q_vec[target] = 1.0
            self.q_vecs[instance_idxs] = labelled_q_vec

        # Initialize iterative updating parameters for q_vecs.
        self.reset_propagation_params()

        # Eventually, we should transparently add a local directory param, so we can spill too disk as necessary.
        # For now, we hold everything in memory.

    def reset_propagation_params(self):
        """
        Helper function to set (or reset) propagation parameters used to track cumulative q_vec values during CCP
        propagation.
        """
        self.propagated_q_vecs = torch.zeros(
            (self.n_unlabelled_targets, self.n_targets), device=DEVICE
        )
        self.propagated_q_vecs_counts = torch.zeros(
            self.n_unlabelled_targets, device=DEVICE
        )

    def reset_q_vecs(self, idxs: List[int]) -> None:
        """
        Reset the q_vecs at the provided idxs.  Reset corresponds to zeroing out all positions.
        :param idxs: A list of indices to reset q-vectors at.
        """
        labelled_idx = next((idx for idx in idxs if idx in self.labelled_idxs), None)
        if labelled_idx is not None:
            raise ValueError(
                f"Cannot reset q-vector for a labelled sample {labelled_idx}!"
            )
        self.q_vecs[idxs] = torch.zeros(self.n_targets, device=DEVICE)

    def update_q_vec(self, idx, q_vec) -> None:
        """
        Saves an updated q vector for the specified idx.
        :param idx: The idx of the q-vector to update.
        :param q_vec: The updated q-vector value.
        """
        if idx in self.labelled_idxs:
            raise ValueError(f"Cannot update q-vector for a labelled sample at {idx}!")
        self.q_vecs[idx] = q_vec

    def get_q_vec(self, idx) -> torch.Tensor:
        """
        Get the q vector for the specified idx.
        :param idx: The idx of the q-vector to return.
        :return: The q-vector value.
        """
        return self.q_vecs[idx]

    def write_q_vecs(self, output_dir: str, filename: str = "q_vecs") -> str:
        """
        Write the current q-vectors to a `.pt` file.
        :param output_dir: The output directory to save q-vectors to.
        :param filename: The desired name of the save file.
        :return: The filepath where the q-vectors were saved.
        """
        if not os.path.isdir(output_dir):
            raise ValueError(f"Cannot find directory {output_dir}!")
        fname = os.path.join(output_dir, f"{filename}.pt")
        torch.save(self.q_vecs, fname)
        return fname

    def load_q_vecs(self, filepath: str) -> None:
        """
        Loads the q-vectors from the saved filepath into the QVectorHandler.
        Also does some validation to make sure the q-vectors are compatible with the QVectorHandler instance.
        :param filepath: The filepath to load q-vectors from.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"Cannot find file {filepath}!")
        q_vecs = torch.load(filepath)
        if q_vecs.shape != (self.n_samples, self.n_targets):
            raise ValueError(
                f"Trying to load q_vecs of the wrong shape! {q_vecs.shape}"
            )
        self.q_vecs = q_vecs

    @staticmethod
    def self_similarity_mask(n: int, n_u: int):
        """
        Create a self-pair similarity mask.

        Although each example appears twice in each batch because of the contrastive sampling,
        we do NOT want to drop the second view of each sample. Transformed views of the same sample
        are considered unique samples (they contain signal and will not be guaranteed to be 1).

        :param n: Number of elements in batch
        :param n_u:  Number of unlabelled elements in batch
        :return:  A boolean mask in which False indicates an identity relation, and True
            indicates that the two examples are NOT transformed from the same underlying
            example - shape <n_u, n>
        """
        self_similarity_mask = torch.logical_not(torch.eye(n))[:n_u, :]  # <n_u, n>
        return self_similarity_mask

    @staticmethod
    def class_evidence(
        unlabelled_z: torch.Tensor, reordered_z: torch.Tensor, reordered_q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class evidences for a transformed batch of samples.  This corresponds to the inner loop of algorithm
        2 for computing psi ψ.

        :param unlabelled_z: The unlabelled transformed samples in the batch.
        :param reordered_z:  The full batch of transformed samples, ordered [unlabelled, labelled].
        :param reordered_q:  The full batch of q vectors, ordered consistently with `reordered_z`.
        :return:  The computed class_evidence (phi) for each transformed element - shape <n_u, c>.
        """
        n_u = unlabelled_z.shape[0]  # Number of unlabelled elements in batch
        n = reordered_z.shape[0]  # Number of elements in batch
        c = reordered_q.shape[1]  # Number of classes (size of q-vecs)

        # Compute z pairwise similarities for unlabelled data:
        z_similarities = pairwise_angular_similarity(
            unlabelled_z, reordered_z
        )  # <n_u, n>

        # Repeat and expand z_similarities and qs for pairwise operation:
        z_similarities_expanded = z_similarities.unsqueeze(1).repeat(
            1, c, 1
        )  # Add an internal axis of size c: <n_u, c, n>
        qs_expanded = reordered_q.T.unsqueeze(0).repeat(
            n_u, 1, 1
        )  # Transpose and add a first axis of size n_u: <n_u, c, n>

        # Create and expand self-pair similarity mask:
        z_self_sim_mask = QVectorHandler.self_similarity_mask(n, n_u)
        z_self_sim_mask_expanded = z_self_sim_mask.unsqueeze(1).repeat(
            1, c, 1
        )  # Add an internal axis of size c: <n_u, c, n>

        # Drop self-pairs and reshape:
        z_similarities_expanded = z_similarities_expanded[
            z_self_sim_mask_expanded
        ].reshape(
            n_u, c, n - 1
        )  # <n_u, c, n-1>
        qs_expanded = qs_expanded[z_self_sim_mask_expanded].reshape(
            n_u, c, n - 1
        )  # <n_u, c, n-1>

        sim_weighted_qs = z_similarities_expanded * qs_expanded  # <n_u, c, n-1>

        # Note: Brody is clipping his un-normalized phi and qs before computing class evidence, but I've convinced
        # myself this is unnecessary because angular similarity is already bounded on the range [0, 1] and q vecs are
        # similarly clipped [0,1] so products will always fall in [0,1] range.
        # There is also no mention of this step in the paper.
        return torch.sum(sim_weighted_qs, dim=2) / torch.sum(
            qs_expanded, dim=2
        )  # Normalize by qs - <n_u, c>

    @staticmethod
    def q_credibility_adjustment(q_vecs):
        """
        Credibility adjustment - subtract the highest similarity from among all other classes.
        :param q_vecs: A 2-D tensor of q vectors to credibility adjust.  Dimensions should be <n, c> for n samples
            with c targets.
        :return: The credibility adjusted q vectors.
        """
        n = q_vecs.shape[0]
        c = q_vecs.shape[1]

        q_vecs_expanded = q_vecs.unsqueeze(1).repeat(1, c, 1)  # <n, c, c>
        self_mask = (
            torch.logical_not(torch.eye(c)).unsqueeze(0).repeat(n, 1, 1)
        )  # <n, c, c>
        # Remove self-pairs so we can get max element not considering self:
        q_vecs_expanded = q_vecs_expanded[self_mask].reshape(n, c, c - 1)  # <n, c, c-1>
        next_class_max_evidence = torch.amax(q_vecs_expanded, dim=2)  # <n, c>
        return q_vecs - next_class_max_evidence

    @staticmethod
    def kl_subsample(
        q_vecs: torch.Tensor, sort_idxs: IdxMask, metadata: CCPMetadata
    ) -> Tuple[float, IdxMask]:
        """
        Compute a maximum subsampling percentage using KL divergence from un-sampled distribution.
        Concretely, we search for the maximum drop % p (in increments of 1%) such that:
        1. Setting the bottom p % of q_vecs to 0 (ordered by sort_idxs, ie magnitude) results in a KL-divergence
            from the un-sampled q_vecs distribution less than metadata.d_max.
        2. p does not increase from the previous iteration (metadata.p_last).
        :param q_vecs: The q-vectors to subsample - expected shape of <n, self.n_targets>.
        :param sort_idxs: An ordering of the 0-dimension of `q_vecs` by magnitude in descending order.  Ie- the first
            element is the idx of the largest element in `q_vecs`, the second idx is the second largest, etc...
        :param metadata: The CCPMetadata associated with the previous CCP iteration (used to threshold the maximum
            allowable KL-divergence, and the maximum p).
        :return: The best discovered p drop (as a float) and an IdxMask representing the q_vec idxs to zero out.
        """
        q_distro = q_vecs.sum(dim=0) / q_vecs.sum()  # Anchor distribution
        n_idxs = sort_idxs.shape[0]
        empty_mask: IdxMask = np.array([])  # type: ignore  # Safe return idx mask
        # We want to select the maximum p_drop such that KL(P | Q) < d_max; it doesn't matter
        # how the KL divergences behave, because we want only the max p -- so we start
        # a sweep from top and exit early:
        max_p = metadata.p_last - QVectorHandler.P_SHIFT
        if max_p <= 0.0:
            LOGGER.warning(
                f"Cannot subsample - previous drop % is {metadata.p_last} and is not allowed to increase!"
            )
            return metadata.p_last, empty_mask
        if max_p >= 1:
            raise ValueError(f"p is a rate; p cannot be {max_p}")
        threshold_idx = n_idxs - int(
            # The next position idx from sort_idxs that won't be zeroed out;
            # as long as max_p is < 1 the threshold_idx will be >=0
            max_p
            * n_idxs
        )
        max_class_weights = q_vecs[sort_idxs[0:threshold_idx]].sum(dim=0)
        max_total_mass = q_vecs[sort_idxs[0:threshold_idx]].sum()

        for p_candidate in [
            i / 100.0 for i in range(int(max_p * 100.0), -1, -1)
        ]:  # -1 as end to include 0 in search
            # Next threshold idx computation:
            next_threshold_idx = n_idxs - int(
                p_candidate * n_idxs
            )  # The next position idx from sort_idxs that won't be zeroed out

            # Update running values for next iteration:
            max_class_weights += q_vecs[
                sort_idxs[threshold_idx:next_threshold_idx]
            ].sum(dim=0)
            max_total_mass += q_vecs[sort_idxs[threshold_idx:next_threshold_idx]].sum()
            threshold_idx = next_threshold_idx

            p_distro = max_class_weights / max_total_mass

            # Compute KL, and exit if divergence is suitably small:
            d_candidate = entropy(
                pk=p_distro.cpu().numpy(),
                qk=q_distro.cpu().numpy(),
                base=2,
            )

            if d_candidate < metadata.d_max:
                # Note this condition is guaranteed to trigger if we ever reach p_candidate == 0.0, so
                # no need for special handling outside of loop.
                LOGGER.info(
                    f"Found suitable subsampling regime at {p_candidate:0.2%} dropped (KL={d_candidate:0.5f})."
                )
                return p_candidate, sort_idxs[threshold_idx:]
        return metadata.p_last, empty_mask

    def propagate_batch_q_vecs(
        self,
        z: torch.Tensor,
        q: torch.Tensor,
        targets: torch.Tensor,
        idxs: torch.Tensor,
    ):
        """
        Function for batch-level q-vector propagation.
        Given a transformed batch, computes the propagated q-vectors.

        Note that all inputs are "stacked" tensors - the z input contains samples passed through transform 1, and then
        samples passed through transform 2.  q, targets and idxs are just duplicated (ie [q1,q2,... ,q1,q2,...]) because
        the transforms are only applied to the samples.  Thus each "batch" contains 2 rows per sample.
        In the documentation below:
        - n is the number of elements in the batch (2x the number of unique samples)
        - z is the output dimension of the projection head z
        - c is the number of target classes (self.n_targets)
        - n_u (n_l) is th number of unlabelled (labelled) element in the batch (2x the number of unique
            unlabelled samples)

        :param z: Tensor of projections of transformed samples for the batch (shape <n, z>) - ie
            [z(t1(s1)), z(t1(s2)), ..., z(t2(s1)), z(t2(s2)), ...]
        :param q: Tensor of q vectors corresponding to elements in batch (shape <n, c>) - ie
            [q1, q2, ..., q1, q2, ...]
        :param targets: Tensor of targets corresponding to elements in batch (shape <n>) - ie
            [0, -1, 2, ..., 0, -1, 2, ...]
        :param idxs: Tensor of idxs corresponding to idx in *overall* dataset of SAMPLES in batch (shape <s>) - ie
            [4531, 3256, 121, ...]
            Note this parameter is expected to be half as long as the others, since it does not get replicated to
            account for transformations.
        """
        # fmt: off
        unlabelled_batch_idxs = (targets == self.unlabelled_target)
        # fmt: on
        labelled_batch_idxs = torch.logical_not(unlabelled_batch_idxs)
        s = idxs.shape[
            0
        ]  # Number of unique samples in batch (Half the number of elements n)

        # Re-order z and q to stack by unlabelled / labelled data.
        unlabelled_z = z[
            unlabelled_batch_idxs
        ]  # Note this will still be in order [z(t1(s1)),  ..., z(t2(s1)), ...]
        reordered_z = torch.cat((unlabelled_z, z[labelled_batch_idxs]), dim=0)
        reordered_q = torch.cat(
            (q[unlabelled_batch_idxs], q[labelled_batch_idxs]), dim=0
        )

        # Compute class evidences:
        class_evidence = self.class_evidence(
            unlabelled_z=unlabelled_z, reordered_z=reordered_z, reordered_q=reordered_q
        )
        credibility_adjusted = self.q_credibility_adjustment(class_evidence)

        # Average over t1 / t2:
        transform_averaged_qs = (
            torch.stack(torch.vsplit(credibility_adjusted, 2), dim=0).sum(dim=0) / 2.0
        )  # <n_u // 2, c>

        # Push these to appropriate index cumulative sums.
        # unlabelled_idxs = idxs[unlabelled_batch_idxs[:s]].to(
        #     int
        # )  # Overall dataset idxs.
        # Find unlabelled_idx positions into propagation subset:
        prop_idxs = torch.searchsorted(
            torch.tensor(self._sorted_target_idxs[self.unlabelled_target]).to(DEVICE),
            idxs[unlabelled_batch_idxs[:s]],
        )
        self.propagated_q_vecs[prop_idxs] += transform_averaged_qs
        self.propagated_q_vecs_counts[prop_idxs] += torch.ones(
            transform_averaged_qs.shape[0], device=DEVICE
        )

    def propagate_q_vecs(self, metadata: CCPMetadata) -> CCPMetadata:
        """
        Complete propagation of q-vectors by updating the soft pseudo-labels.
        Q-vector propagation averages across all epochs to combat self-reinforcing
        false confidence; "all epochs" values is tracked by self.propagated_q_vecs and
        self.propagated_q_vecs_counts.
        1. Scale and normalize cumulative sum q vectors.
        2. Subsample according to KL-divergence and provided metadata.
        3. Clip and credibility adjust, and set unlabelled q-vectors to new propagated values.
        :param metadata: The metadata from the previous CCP iteration.
        :return: CCPMetadata for next iteration of CCP.
        """
        # Sanity-check to make sure at least a full epoch has elapsed.
        if not all(self.propagated_q_vecs_counts):  # Validate that all positions > 0
            raise RuntimeError(
                f"Cannot propagate q vectors without every unlabelled sample represented at least once!"
            )

        # Scale and average propagating q vecs:
        scale_factor = torch.max(self.propagated_q_vecs)
        self.propagated_q_vecs /= self.propagated_q_vecs_counts.unsqueeze(
            1
        )  # Average cumulative sum by count

        # Scale strongest signal to 1.0:
        if scale_factor > 0.0:
            # fmt: off
            self.propagated_q_vecs *= (1.0 / scale_factor)
            # fmt: on
        else:
            LOGGER.warning(
                f"Failed to scale q-vectors - largest magnitude is <= 0.0: {scale_factor}"
            )

        # Reset propagation counts:
        self.propagated_q_vecs_counts = torch.zeros(
            self.n_unlabelled_targets, device=DEVICE
        )

        # Get un-clipped, un-adjusted max weights for propagated q vectors (to rank subsampling):
        sort_idxs = (
            torch.amax(self.propagated_q_vecs, dim=1)
            .argsort(descending=True)
            .cpu()
            .numpy()
        )  # <n_u>
        # Clip and cred-adjust propagated q-vectors:
        self.propagated_q_vecs = torch.clamp(
            self.q_credibility_adjustment(self.propagated_q_vecs), min=0.0, max=1.0
        )

        # Compute subsamples:
        p_drop, idx_to_drop = self.kl_subsample(
            self.propagated_q_vecs, sort_idxs, metadata
        )
        self.propagated_q_vecs[idx_to_drop] = torch.zeros(
            idx_to_drop.shape[0], self.n_targets, device=DEVICE
        )

        # Propagate q vectors:
        LOGGER.info(f"Updating q vectors for unlabelled targets...")
        self.q_vecs[
            self._sorted_target_idxs[self.unlabelled_target]
        ] = self.propagated_q_vecs

        # Reset propagation parameters:
        self.reset_propagation_params()
        return CCPMetadata(
            p_last=p_drop, d_max=metadata.d_max * self.KL_DIVERGENCE_MAX_SCALING_FACTOR
        )
