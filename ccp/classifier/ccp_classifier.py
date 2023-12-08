import itertools
import logging
from typing import Callable, Dict, Optional, Tuple, Type

import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from ccp.classifier.classifier_loss import SoftCrossEntropyLoss
from ccp.classifier.q_label_dataset import QLabelDataset
from ccp.device_decision import DEVICE
from ccp.training import EMALossExitCriteria, ema_training_loop

LOGGER = logging.getLogger(__name__)


class ContrastiveCredibilityClassifier(object):
    def __init__(
        self,
        encoder_network_f_b: nn.Module,
        projection_head_f_g: nn.Module,
        q_dataset: QLabelDataset,
        batch_size: int = 256,
        # Network Initialization Parameters:
        network_init_func: Optional[Callable] = None,
        # Optimization Parameters:
        optimizer: Type[optim.Optimizer] = optim.Adam,
        optimizer_parameters: Dict = {
            "lr": 0.00004,
            "weight_decay": 0.004,
        },  # Weight decay is l2_regularization
    ):
        """
        A classifier that can operate on top of q-vectors rather than singleton labels.

        :param encoder_network_f_b: An encoder that may be prewarmed (paper recommends to prewarm the encoder)
        :param projection_head_f_g: A projection head for classification to attach to the encoder output.  The paper
            uses a simple 2-layer MLP:
            nn.Sequential(
               nn.Linear(encoder_output_dim, projection_hidden_dim),
               nn.ReLU(),
               nn.Linear(projection_hidden_dim, num_targets),
            )
            Input dimension must be compatible with encoder, and output dimension must be the number of targets.
        :param q_dataset: A QLabelDataset Instance to train on.
        :param batch_size: The size of the data batches to operate on.
        """
        self.encoder = encoder_network_f_b
        self.projection_head = projection_head_f_g

        self.dataloader = DataLoader(q_dataset, batch_size=batch_size, shuffle=True)

        if network_init_func:
            LOGGER.info(
                "Re-initializing networks parameters using provided initialization function!"
            )
            for module in [self.encoder, self.projection_head]:
                module.apply(network_init_func)

        self.optimizer = optimizer(
            params=itertools.chain(
                self.encoder.parameters(), self.projection_head.parameters()
            ),
            **optimizer_parameters,
        )

        self.criterion = SoftCrossEntropyLoss()

    def train_classifier_single_epoch(
        self, print_loss_every_k_batches: int
    ) -> Tuple[float, Dict]:
        """
        Train classifier for a single epoch.

        :param print_loss_every_k_batches: How frequently to average and print the cumulative loss.
        :return: Average loss on the epoch and a set of default args with which to call this method
            next time
        """
        cumulative_loss = 0.0
        batch_count = 0
        for b_idx, (samples, q_vecs) in enumerate(self.dataloader):
            samples = samples.to(DEVICE)
            q_vecs = q_vecs.to(DEVICE)

            self.optimizer.zero_grad()
            batch_g = self.projection_head(self.encoder(samples))
            loss = self.criterion(batch_g, q_vecs)
            loss.backward()
            self.optimizer.step()

            cumulative_loss += loss.item()
            batch_count += 1

            # Print loss every `print_loss_every_k_batches` mini-batches
            if b_idx % print_loss_every_k_batches == (print_loss_every_k_batches - 1):
                print(f"[{b_idx + 1:5d}] loss: {cumulative_loss / batch_count:.3f}")

        loss = cumulative_loss / batch_count
        args = {"print_loss_every_k_batches": print_loss_every_k_batches}
        return loss, args

    def ema_train(
        self,
        exit_criteria: EMALossExitCriteria,
        print_loss_every_k_batches: int = 5,
    ):
        """
        Full training loop for learning a classifier using EMA loss as an exit criteria.

        :param exit_criteria: An EMALossExitCriteria instance defining how to compute the EMA loss and when to complete
            training.
        :param print_loss_every_k_batches: How often to print the loss during an epoch of training.
        :returns: The best loss achieved.
        """
        LOGGER.info("Training Soft-label Classifier.")

        epochs, best_loss, _ = ema_training_loop(
            single_iteration_func=self.train_classifier_single_epoch,
            exit_criteria=exit_criteria,
            initial_func_params={
                "print_loss_every_k_batches": print_loss_every_k_batches
            },
        )

        LOGGER.info(
            f"Completed classifier training after {epochs} epochs, with a best EMA loss of {best_loss:0.6f}."
        )
        return best_loss

    def classifier(self) -> nn.Sequential:
        """
        Retrieve the classifier as a nn.Module.
        :return: The current state classifier as a single module.
        """
        return nn.Sequential(self.encoder, self.projection_head)

    @property
    def num_targets(self) -> int:
        """
        Returns the size of the target classification space.
        """
        clf = self.classifier()
        return clf[-1].out_features
