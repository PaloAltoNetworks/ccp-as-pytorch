"""
Unit tests for the data module.
"""
import tempfile
import unittest
from collections import Counter
from itertools import chain

import numpy as np
import pytest
import torch
from scipy.stats import entropy  # type: ignore

from ccp.datareaders import DataReader, SimpleDataReader
from ccp.softlabeler.data import CCPBatchSampler, CCPDataset, QVectorHandler
from ccp.softlabeler.label_loss import pairwise_angular_similarity
from ccp.typing import CCPMetadata


class TestCCPDataset(unittest.TestCase):
    """
    Tests against CCPDataset class.
    """

    # Init class-level vars for mypy.
    X = None
    y = None
    unlab_X = None
    datareader = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.X = torch.rand((10, 20, 2))  # 10 examples of dim 20, 2
        cls.y = np.array([int(i / 3) for i in range(10)])  # 10 labels (0, 1, 2, 3)
        cls.unlab_X = torch.rand((50, 20, 2))  # 50 examples of dim 20, 4
        with tempfile.NamedTemporaryFile() as lab_data, tempfile.NamedTemporaryFile() as unlab_data:
            np.savez_compressed(lab_data, data=cls.X, labels=cls.y)
            np.savez_compressed(unlab_data, data=cls.unlab_X)

            cls.datareader = SimpleDataReader(
                labelled_npz_fpath=lab_data.name, unlabelled_npz_fpath=unlab_data.name
            )

    def test_ccp_datasset(self):
        dataset = CCPDataset(data_reader=self.datareader)
        n_targets = len(np.unique(self.y))
        self.assertEqual(
            self.X.shape[0] + self.unlab_X.shape[0],
            len(dataset),
            "Dataset has length that is equal to labelled + unlabelled data.",
        )

        # Check that we can "get" an item from the dataset, and that it returns a coherent Q vector:
        unlabelled_idx = self.datareader.sorted_target_idxs[
            DataReader.UNLABELLED_TARGET
        ][0]
        _, sample_q, target, idx = dataset[unlabelled_idx]
        np.testing.assert_array_equal(sample_q, np.zeros(n_targets))
        self.assertEqual(unlabelled_idx, idx)
        self.assertEqual(target, DataReader.UNLABELLED_TARGET)

        labelled_idx = self.datareader.sorted_target_idxs[self.y[0]][0]
        _, sample_q, target, idx = dataset[labelled_idx]
        expected_q = np.zeros(n_targets)
        expected_q[0] = 1.0
        np.testing.assert_array_equal(sample_q, expected_q)
        self.assertEqual(labelled_idx, idx)
        self.assertEqual(target, self.y[0])


class TestQVectorHandler(unittest.TestCase):
    q_handler = None
    unlab_idxs = None
    Z_DIM = 16

    @classmethod
    def setUpClass(cls) -> None:
        cls.unlab_idxs = [4, 5]
        cls.q_handler = QVectorHandler(
            n_samples=10,
            sorted_target_idxs={
                -1: np.array([4, 5]),
                0: np.array([0, 1, 2]),
                1: np.array([3, 6]),
                2: np.array([7, 8, 9]),
            },
            unlabelled_target=-1,
        )

    def test_write_load(self):
        # Set and retrieve some q-vectors:
        random_qs = {}
        for idx in self.unlab_idxs:
            random_q = torch.rand((self.q_handler.n_targets,))
            self.q_handler.update_q_vec(idx=idx, q_vec=random_q)
            self.assertTrue(torch.equal(random_q, self.q_handler.get_q_vec(idx=idx)))
            random_qs[
                idx
            ] = random_q  # Save a reference to the random Q to validate against later.

        with tempfile.TemporaryDirectory() as q_vecs_directory:
            # Write to disk:
            filepath = self.q_handler.write_q_vecs(output_dir=q_vecs_directory)

            # Reset q-vectors.
            self.q_handler.reset_q_vecs(idxs=self.unlab_idxs)
            for idx in self.unlab_idxs:
                np.testing.assert_equal(
                    np.zeros(self.q_handler.n_targets),
                    self.q_handler.get_q_vec(idx=idx),
                )

            # Load from disk.
            self.q_handler.load_q_vecs(filepath=filepath)

        # Assert that random q-vectors are reloaded correctly:
        for idx, random_q in random_qs.items():
            self.assertTrue(torch.equal(random_q, self.q_handler.get_q_vec(idx=idx)))

    def test_self_similarity_mask_example(self):
        """
        Test that the self-similarity mask generates the diagonal plus
        additional contrastive pairs, and that any labeled columns are
        indeed used.
        """
        observed_out = self.q_handler.self_similarity_mask(6, 4)
        expected_out = torch.Tensor(
            [
                [False, True, True, True, True, True],
                [True, False, True, True, True, True],
                [True, True, False, True, True, True],
                [True, True, True, False, True, True],
            ]
        )
        self.assertTrue(observed_out.equal(expected_out))

    def test_class_evidence(self):
        """
        Test execution of class_evidence.
        """
        reordered_z = torch.randn(2 * self.q_handler.n_samples, self.Z_DIM)
        unlabelled_z = reordered_z[: 2 * len(self.unlab_idxs)]
        reordered_q = torch.clamp(
            torch.randn(2 * self.q_handler.n_samples, self.q_handler.n_targets),
            min=0.0,
            max=1.0,
        )
        class_evidence = self.q_handler.class_evidence(
            unlabelled_z=unlabelled_z, reordered_z=reordered_z, reordered_q=reordered_q
        )
        self.assertEqual(
            class_evidence.shape, (2 * len(self.unlab_idxs), self.q_handler.n_targets)
        )

    def test_class_evidence_example(self):
        """
        Verify that the vectorization is correct by using an alternative
        method to calculate a value.

        This implementation tests the psi component ψ of Algorithm 2.

        We have 3 target classes.
        We have three examples. What we know BEFORE the cycle (q values) is:
        * j0 looks a little like class 0; its embedding is near (1, 1)
        * j1 is class 1; its embedding is near (-1, -1)
        * j2 looks a lot like class 2; its embedding is near (0, 1)
        We also know that the transformed pairs have similar z-embeddings.
        """
        n_unlabelled = 2
        reordered_z = torch.Tensor(
            [
                [0.9415, 1.1867],  # j=0; unlabeled; t1
                [0.0641, 0.9733],  # j=2; unlabeled; t1
                [0.9729, 1.0952],  # j=0; unlabeled; t2
                [-0.0941, 1.1162],  # j=2; unlabeled; t2
                [-1.1291, -0.9411],  # j=1; labeled as class 1; t1
                [-1.2000, -1.2282],  # j=1; labeled as class 1; t2
            ]
        )
        reordered_q = torch.Tensor(
            [
                [0.3000, 0.1000, 0.050],  # j=0; unlabeled; t1
                [0.0200, 0.0000, 0.4893],  # j=2; unlabeled; t1
                [0.3000, 0.1000, 0.050],  # j=0; unlabeled; t2
                [0.0200, 0.0000, 0.4893],  # j=2; unlabeled; t2
                [0.0000, 1.0000, 0.0000],  # j=1; labeled as class 1; t1
                [0.0000, 1.0000, 0.0000],  # j=1; labeled as class 1; t2
            ]
        )
        unlabelled_z = reordered_z[: 2 * n_unlabelled]
        observed_class_evidence = self.q_handler.class_evidence(
            unlabelled_z=unlabelled_z, reordered_z=reordered_z, reordered_q=reordered_q
        )

        # walk through all the phis we need to calculate - we treat each z as an independent instance, and only avoid
        # comparing a particular element against itself:
        # This tests a mathematical formalism that is from a previous version of the paper (not the version from December 2023)
        for unlabelled_row_t_j in [0, 1, 2, 3]:
            for class_of_interest_k in range(3):
                expected_numerator = 0
                expected_denominator = 0
                for example_row_i in (i for i in range(6) if i != unlabelled_row_t_j):
                    # Grab the q corresponding to this example and class:
                    q_of_interest = reordered_q[example_row_i, class_of_interest_k]

                    # Weights are similarities between relevant rows
                    weight = pairwise_angular_similarity(
                        reordered_z[unlabelled_row_t_j, None],
                        reordered_z[example_row_i, None],
                    ).squeeze()

                    expected_numerator += weight * q_of_interest
                    expected_denominator += q_of_interest

                with self.subTest(
                    f"vectorization mismatch: considering t(j) row={unlabelled_row_t_j}, class {class_of_interest_k}"
                ):
                    self.assertEqual(
                        observed_class_evidence[
                            unlabelled_row_t_j, class_of_interest_k
                        ],
                        expected_numerator / expected_denominator,
                    )

        # assert vectorization makes sense: none of the unlabeled examples
        # is like the labeled example in class 1
        assert observed_class_evidence.argmin(dim=1).tolist() == [1, 1, 1, 1]

    def test_cred_adjust_argmax(self):
        """
        Test that argmax does not change under credibility adjustment.
        """
        q = torch.clamp(
            torch.randn(2 * self.q_handler.n_samples, self.q_handler.n_targets),
            min=0.0,
            max=1.0,
        )
        cred_adjust_q = self.q_handler.q_credibility_adjustment(q_vecs=q)
        # torch.argmax handles multiple identical maximum values consistently
        # (always returns the first)
        self.assertTrue(
            torch.equal(torch.argmax(q, dim=1), torch.argmax(cred_adjust_q, dim=1))
        )

    def test_cred_adjust_example(self):
        """
        Test credibility_adjustment with an example.
        """
        q_vec_input = torch.Tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.05]])
        expected_out = torch.Tensor([[-0.02, -0.01, 0.01], [-0.01, 0.00, 0.00]])
        observed_out = self.q_handler.q_credibility_adjustment(q_vecs=q_vec_input)
        self.assertTrue(expected_out.isclose(observed_out).all())

    def test_kl_subsample(self):
        """
        Test execution of kl_subsample.
        """
        #  q_vecs: torch.Tensor, sort_idxs: IdxMask, metadata: CCPMetadata
        q = torch.randn(150, self.q_handler.n_targets)
        sort_idxs = torch.amax(q, dim=1).argsort()
        metadata = CCPMetadata(p_last=0.9, d_max=0.01)
        q = torch.clamp(q, min=0.0, max=1.0)

        p, drop_idxs = self.q_handler.kl_subsample(
            q_vecs=q, sort_idxs=sort_idxs, metadata=metadata
        )
        drop_count = q[drop_idxs].shape[0]
        self.assertEqual(drop_count, int(p * q.shape[0]))
        # Also check that with a max_p of 0.0 we exit out early:
        p, drop_idxs = self.q_handler.kl_subsample(
            q_vecs=q, sort_idxs=sort_idxs, metadata=CCPMetadata(p_last=0.0, d_max=0.01)
        )
        drop_count = q[drop_idxs].shape[0]
        self.assertEqual(drop_count, int(p * q.shape[0]))
        self.assertEqual(drop_idxs.shape[0], 0)  # Should be empty

    def test_kl_subsample_example(self):
        """
        Test an example executing kl_subsample.

        Given an example of 4 unlabeled data, where 3 q-vectors have a tiny impact
        on the overall distribution of classes, and one q-vector has an outsized impact,
        we'd like to see that the KL-divergence tells us that we can achieve
        pretty low divergence by excluding all 3 of the tiny impact vectors.
        """
        q_vecs = torch.Tensor(
            [
                [0.34, 0.3, 0.3],  # tiny
                [0.3, 0.34, 0.3],  # tiny
                [0.3, 0.3, 0.34],  # tiny
                [0.95, 0.01, 0.01],
            ]  # massively informative about relative class distribution
        )
        sort_idxs = (
            torch.amax(q_vecs, dim=1)
            .argsort(descending=True)  # descending per propagate_q_vecs
            .numpy()
        )
        q_vecs = torch.clamp(
            self.q_handler.q_credibility_adjustment(q_vecs), min=0.0, max=1.0
        )

        # Verify that if we include the single most informative example,
        # we instantly achieve a reasonably low level of divergence -- in other
        # words, that our drop_idxs are all the other examples
        p_start = 0.9
        entropy_from_heaviest_example = entropy(
            pk=np.array([1, 0.0, 0.0]), qk=q_vecs.sum(dim=0) / q_vecs.sum(), base=2
        )
        p_new, drop_idxs = self.q_handler.kl_subsample(
            q_vecs=q_vecs,
            sort_idxs=sort_idxs,
            metadata=CCPMetadata(
                p_last=p_start, d_max=entropy_from_heaviest_example + 0.001
            ),
        )
        self.assertSetEqual(set(drop_idxs), set([0, 1, 2]))
        self.assertEqual(p_new, p_start - self.q_handler.P_SHIFT)

    def test_prop_batch(self):
        """
        Test execution of batch level propagation.
        """
        z = torch.randn(2 * self.q_handler.n_samples, self.Z_DIM)
        q = torch.clamp(
            torch.randn(2 * self.q_handler.n_samples, self.q_handler.n_targets),
            min=0.0,
            max=1.0,
        )

        targets = torch.ones(self.q_handler.n_samples)
        idxs = torch.ones(self.q_handler.n_samples)
        for target, t_idxs in self.q_handler._sorted_target_idxs.items():
            for t_idx in t_idxs:
                targets[t_idx] = target
                idxs[t_idx] = t_idx

        self.q_handler.propagate_batch_q_vecs(
            z=z, q=q, targets=targets.repeat(2), idxs=idxs
        )

    @pytest.mark.filterwarnings(
        "ignore:  Subsampling step is too large"
    )  # Expected warning for kl divergence given small unlabelled count
    def test_prop(self):
        """
        Test execution of overall q-vec propagation.
        """
        metadata = CCPMetadata(p_last=0.9, d_max=0.01)
        # Assert that call fails if no propagated values exist:
        self.assertRaises(
            RuntimeError, lambda: self.q_handler.propagate_q_vecs(metadata)
        )

        # Test normal flow - first run some batches through the system:
        for _ in range(5):
            z = torch.randn(2 * self.q_handler.n_samples, self.Z_DIM)
            q = torch.clamp(
                torch.randn(2 * self.q_handler.n_samples, self.q_handler.n_targets),
                min=0.0,
                max=1.0,
            )

            targets = torch.ones(self.q_handler.n_samples)
            idxs = torch.ones(self.q_handler.n_samples)
            for target, t_idxs in self.q_handler._sorted_target_idxs.items():
                for t_idx in t_idxs:
                    targets[t_idx] = target
                    idxs[t_idx] = t_idx

            self.q_handler.propagate_batch_q_vecs(
                z=z, q=q, targets=targets.repeat(2), idxs=idxs
            )

        # Now normalize and propagate:
        self.q_handler.propagate_q_vecs(metadata=CCPMetadata(p_last=0.9, d_max=0.01))


class TestCCPBatchSampler(unittest.TestCase):
    @property
    def mock_dataset_target_idxs(self):
        return {
            -1: list(range(10)),
            0: list(range(10, 15)),
            1: list(range(15, 20)),
            2: list(range(20, 35)),
            3: list(range(35, 40)),
        }

    @property
    def mock_dataset_idx_target(self):
        return {
            **dict.fromkeys(list(range(10)), -1),
            **dict.fromkeys(list(range(10, 15)), 0),
            **dict.fromkeys(list(range(15, 20)), 1),
            **dict.fromkeys(list(range(20, 35)), 2),
            **dict.fromkeys(list(range(35, 40)), 3),
        }

    def test_failure_modes(self):
        # Test that batch_size < number of targets raises an error:
        self.assertRaises(
            ValueError,
            lambda: CCPBatchSampler(
                batch_size=2, dataset_target_idxs=self.mock_dataset_target_idxs
            ),
        )

        # Test that batch_size < number of targets at sample rate raises an error:
        self.assertRaises(
            ValueError,
            lambda: CCPBatchSampler(
                batch_size=10,
                dataset_target_idxs=self.mock_dataset_target_idxs,
                target_sample_rates={0: 10},
            ),
        )

        # Test that unmapped epoch_target_cond raises an error:
        self.assertRaises(
            ValueError,
            lambda: CCPBatchSampler(
                batch_size=10,
                dataset_target_idxs=self.mock_dataset_target_idxs,
                epoch_target_cond=15,
            ),
        )

        # Test that accessing `batches` before calling `generate_batches` raises an Error:
        sampler = CCPBatchSampler(
            batch_size=10, dataset_target_idxs=self.mock_dataset_target_idxs
        )
        self.assertRaises(ValueError, lambda: sampler.batches)
        self.assertRaises(ValueError, lambda: len(sampler))

    def test_batch_sampler(self):
        largest_target = max(
            self.mock_dataset_target_idxs.items(), key=lambda kv: len(kv[1])
        )[0]

        # Simple sampler case (batch size = number of targets, no upsampling)
        sampler = CCPBatchSampler(
            batch_size=len(self.mock_dataset_target_idxs),
            dataset_target_idxs=self.mock_dataset_target_idxs,
        )

        batches = list(iter(sampler))
        # With balanced sampling and batches = number of targets number of batches should equal single largest target:
        self.assertEqual(
            len(batches), len(self.mock_dataset_target_idxs[largest_target])
        )

        # Test that all idx are present in batches - and that for largest target they only appear once:
        idx_counter = Counter()
        for batch in batches:
            idx_counter.update(batch)

        self.assertTrue(
            all(
                idx in idx_counter
                for idx in chain.from_iterable(self.mock_dataset_target_idxs.values())
            )
        )
        self.assertTrue(
            all(
                idx_counter[idx] == 1
                for idx in self.mock_dataset_target_idxs[largest_target]
            )
        )

        # Test that each batch contains at least one of each target:
        self.assertTrue(
            all(
                len({self.mock_dataset_idx_target[idx] for idx in batch})
                == len(self.mock_dataset_target_idxs)
                for batch in batches
            )
        )

        # Test `epoch_target_cond`:
        epoch_target_cond = 0
        sampler = CCPBatchSampler(
            batch_size=len(self.mock_dataset_target_idxs),
            dataset_target_idxs=self.mock_dataset_target_idxs,
            epoch_target_cond=epoch_target_cond,
        )
        batches = list(iter(sampler))
        # With balanced sampling, batches = number of targets, an epoch_target_cond number of batches should equal
        # epoch_target_cond:
        self.assertEqual(
            len(batches), len(self.mock_dataset_target_idxs[epoch_target_cond])
        )

        # Test that at most 1 epoch_cond target sample appears twice:
        idx_counter = Counter()
        for batch in batches:
            idx_counter.update(batch)

        # Assert that count of idx appearing more than once is <= 1:
        self.assertTrue(
            [
                idx_counter[idx] > 1
                for idx in self.mock_dataset_target_idxs[epoch_target_cond]
            ].count(True)
            <= 1
        )

        # With upscaling, we get non-deterministic bath numbers depending on the randomly sampled target -
        # but validate that each batch contains at least the right number of target samples:

        # Test upsampling:
        target_sample_rates = {0: 4, 2: 2}
        sampler = CCPBatchSampler(
            batch_size=2 * len(self.mock_dataset_target_idxs),
            dataset_target_idxs=self.mock_dataset_target_idxs,
            target_sample_rates=target_sample_rates,
        )
        batches = list(iter(sampler))
        batch_target_count = [
            Counter([self.mock_dataset_idx_target[idx] for idx in batch])
            for batch in batches
        ]
        self.assertTrue(
            all(
                all(
                    counter[target]
                    >= (
                        target_sample_rates[target]
                        if target in target_sample_rates
                        else 1
                    )
                    for target in self.mock_dataset_target_idxs
                )
                for counter in batch_target_count
            )
        )

        # Test upsampling and `epoch_target_cond`:
        sampler = CCPBatchSampler(
            batch_size=2 * len(self.mock_dataset_target_idxs),
            dataset_target_idxs=self.mock_dataset_target_idxs,
            epoch_target_cond=epoch_target_cond,
            target_sample_rates=target_sample_rates,
        )

        batches = list(iter(sampler))
        batch_target_count = [
            Counter([self.mock_dataset_idx_target[idx] for idx in batch])
            for batch in batches
        ]
        self.assertTrue(
            all(
                all(
                    counter[target]
                    >= (
                        target_sample_rates[target]
                        if target in target_sample_rates
                        else 1
                    )
                    for target in self.mock_dataset_target_idxs
                )
                for counter in batch_target_count
            )
        )
        # Test that at most 1 epoch_cond target sample appears twice:
        idx_counter = Counter()
        for batch in batches:
            idx_counter.update(batch)

        # Assert that count of idx appearing more than once is <= 1:
        self.assertTrue(
            [
                idx_counter[idx] > 1
                for idx in self.mock_dataset_target_idxs[epoch_target_cond]
            ].count(True)
            <= 1
        )
