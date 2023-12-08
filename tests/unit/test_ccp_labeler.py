import tempfile
import unittest

import torch

from ccp.datareaders import DataReader
from ccp.encoders.ccp_text_encoder import TextEncoder
from ccp.softlabeler.ccp_labeler import ContrastiveCredibilityLabeller


class MockDataReader(DataReader):
    @property
    def sorted_target_idxs(self):
        return {DataReader.UNLABELLED_TARGET: [], 0: [0]}

    def __getitem__(self, idx: int):
        return ["mock"], 0


class MockTransform(object):
    def __call_(self):
        return None


class TestContrastiveCredibilityLabeller(unittest.TestCase):
    """
    Tests against ContrastiveCredibilityLabeller
    """

    def test_initialization_propagation(self):
        """
        Verify that initialization recursively sets weights and biases throughout
        both the encoder network f_b and the projection head f_z.
        """

        def _get_first_scalar(param: torch.nn.parameter.Parameter) -> float:
            dimensionality = len(param.shape)
            if dimensionality == 4:
                element = param[0, 0, 0, 0].item()
            elif dimensionality == 1:
                element = param[0].item()
            else:
                raise NotImplementedError(
                    f"Unexpected parameter shape: {dimensionality}"
                )
            return element

        def _set_first_scalar(param: torch.nn.parameter.Parameter, target_value: float):
            dimensionality = len(param.shape)
            if dimensionality == 4:
                param.data[0, 0, 0, 0] = target_value
            elif dimensionality == 1:
                param.data[0] = target_value
            else:
                raise NotImplementedError(
                    f"Unexpected parameter shape: {dimensionality}"
                )

        with tempfile.TemporaryDirectory() as tempdir_name:
            ccp_labeler = ContrastiveCredibilityLabeller(
                data_reader=MockDataReader(),
                output_dir=tempdir_name,
                transforms=[MockTransform()],
                encoder_network_f_b=TextEncoder(dim_in=(10, 10)),
                projection_head_f_z=torch.nn.Linear(10, 5),
                batch_size=1,
                target_sample_rates={DataReader.UNLABELLED_TARGET: 0},
            )

            # Set a cell to a canary value and verify the value persisted
            CANARY_VALUE = 99
            for name, param in ccp_labeler.encoder.named_parameters():
                with self.subTest(
                    f"Verifying preconditions and setting {name} to a canary value"
                ):
                    self.assertNotEqual(_get_first_scalar(param), CANARY_VALUE)
                    _set_first_scalar(param, CANARY_VALUE)
                    self.assertEqual(_get_first_scalar(param), CANARY_VALUE)

            # Reinitialize and verify the canary value is gone
            ccp_labeler.randomly_initialize_parameters()
            for name, param in ccp_labeler.encoder.named_parameters():
                with self.subTest(
                    f"Verifying that reset of parameters succeeded for {name}"
                ):
                    self.assertNotEqual(_get_first_scalar(param), CANARY_VALUE)
