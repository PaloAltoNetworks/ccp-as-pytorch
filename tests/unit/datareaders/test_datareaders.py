import tempfile
import unittest

import numpy as np
import torch

from ccp.datareaders import DataReader, SimpleDataReader


class TestDataReader(unittest.TestCase):
    """
    Tests against abstract DataReader base class.
    """

    def test_validate_target_idxs(self):
        invald_inputs = [
            {0: ["not", "numeric"], 1: [0, 1, 2, 3]},
            {0: [0, 1, 2], 2: [3, 4]},
            {DataReader.UNLABELLED_TARGET: [4], 0: [0, 1, 2, 3, 4]},
            {DataReader.UNLABELLED_TARGET: [-1, 4], 0: [2, 3]},
            {DataReader.UNLABELLED_TARGET: [0, 1, 2, 2], 0: [3, 3, 4, 5]},
            {DataReader.UNLABELLED_TARGET: [2, 1, 0], 0: [3, 4, 5]},
            {DataReader.UNLABELLED_TARGET: [0, 1, 2], 0: [3, 4, 8]},
        ]
        for invalid_input in invald_inputs:
            self.assertRaises(
                ValueError, lambda: DataReader._validate_target_idxs(invalid_input)
            )

        valid_inputs = [
            {DataReader.UNLABELLED_TARGET: [1, 2, 3], 0: [4, 5, 6], 1: [0, 7, 8]}
        ]
        for valid_input in valid_inputs:
            DataReader._validate_target_idxs(valid_input)  # Shouldn't raise any errors


class TestSimpleDataReader(unittest.TestCase):
    def test_failure_modes(self):
        # Non-existent input files should raise an error:
        self.assertRaises(
            ValueError,
            lambda: SimpleDataReader(
                labelled_npz_fpath="not-a-file", unlabelled_npz_fpath="not-a-file"
            ),
        )

        with tempfile.NamedTemporaryFile() as lab_data, tempfile.NamedTemporaryFile() as unlab_data:
            # Mismatch in label-shape and inputs should raise an error:
            X = torch.rand((10, 20))  # 10 examples of dim 20
            y = torch.ones((15,))  # 15 labels
            np.savez_compressed(lab_data, data=X, labels=y)
            self.assertRaises(
                ValueError,
                lambda: SimpleDataReader(
                    labelled_npz_fpath=lab_data.name,
                    unlabelled_npz_fpath=unlab_data.name,
                ),
            )

            # DataReader.UNLABELLED_TARGET in  label set should raise an error:
            y = torch.ones(10)
            y[0] = DataReader.UNLABELLED_TARGET
            np.savez_compressed(lab_data, data=X, labels=y)
            self.assertRaises(
                ValueError,
                lambda: SimpleDataReader(
                    labelled_npz_fpath=lab_data.name,
                    unlabelled_npz_fpath=unlab_data.name,
                ),
            )

            # Size mismatch between labelled/unlabelled data should raise an error:
            y = torch.ones(10)
            np.savez_compressed(lab_data, data=X, labels=y)

            unlab_X = torch.rand((30, 20, 4))  # 10 examples of dim (20, 4)
            np.savez_compressed(unlab_data, data=unlab_X)
            self.assertRaises(
                ValueError,
                lambda: SimpleDataReader(
                    labelled_npz_fpath=lab_data.name,
                    unlabelled_npz_fpath=unlab_data.name,
                ),
            )

    def test_simple_datareader(self):
        with tempfile.NamedTemporaryFile() as lab_data, tempfile.NamedTemporaryFile() as unlab_data:
            X = torch.rand((10, 20, 2))  # 10 examples of dim 20, 2
            y = np.array([int(i / 3) for i in range(10)])  # 10 labels (0, 1, 2, 3)
            np.savez_compressed(lab_data, data=X, labels=y)

            unlab_X = torch.rand((50, 20, 2))  # 50 examples of dim 20, 4
            np.savez_compressed(unlab_data, data=unlab_X)

            datareader = SimpleDataReader(
                labelled_npz_fpath=lab_data.name, unlabelled_npz_fpath=unlab_data.name
            )

            self.assertEqual(
                set(y) | {DataReader.UNLABELLED_TARGET},
                datareader.sorted_target_idxs.keys(),
                "All labels should be mapped to indices in the dataset.classes property!",
            )

            self.assertEqual(datareader.n_labelled, X.shape[0])
            self.assertEqual(datareader.n_unlabelled, unlab_X.shape[0])

            # Check that we can "get" an item from the data reader:
            unlabelled_idx = datareader.sorted_target_idxs[
                DataReader.UNLABELLED_TARGET
            ][0]
            sample, target = datareader.__getitem__(unlabelled_idx)

            np.testing.assert_equal(sample, unlab_X[(unlabelled_idx - X.shape[0]), :])
            self.assertEqual(target, DataReader.UNLABELLED_TARGET)

            labelled_idx = datareader.sorted_target_idxs[y[0]][0]
            sample, target = datareader.__getitem__(labelled_idx)

            np.testing.assert_array_equal(sample, X[labelled_idx, :])
            self.assertEqual(target, y[0])
