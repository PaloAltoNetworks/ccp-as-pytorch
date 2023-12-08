import copy
import itertools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from ccp.ccp_training_options import CCPRegime, CCPTrainingOptions
from ccp.classifier.q_label_dataset import QLabelDataset
from ccp.datareaders import DataReader
from ccp.device_decision import DEVICE
from ccp.param_init import init_weights_ccp
from ccp.softlabeler.data.ccp_batch_sampler import CCPBatchSampler
from ccp.softlabeler.data.ccp_dataset import CCPDataset
from ccp.softlabeler.label_loss import SoftSupervisedContrastiveLoss
from ccp.softlabeler.transforms.transform_apply import TransformApply
from ccp.training import EMALossExitCriteria, ema_training_loop
from ccp.typing import CCPMetadata, TargetLabel

LOGGER = logging.getLogger(__name__)


class ContrastiveCredibilityLabeller(object):
    def __init__(
        self,
        data_reader: DataReader,
        output_dir: str,
        encoder_network_f_b: nn.Module,
        projection_head_f_z: nn.Module,
        # Training Options (Optimizer, Learning Schedule):
        training_options: CCPTrainingOptions,
        transforms: List[Callable],
        independently_transform_samples: bool = True,
        batch_size: int = 256,
        target_sample_rates: Dict[TargetLabel, int] = {
            DataReader.UNLABELLED_TARGET: 10
        },
        random_seed: int = 42,
        # Network Initialization Parameters:
        network_init_func: Callable = init_weights_ccp,
    ):
        """
        Entry point class for performing contrastive credibility propagation. This class
        learns the soft label q-vector representation for the unlabeled data in `dataset`.

        :param data_reader: A concrete implementation of the DataReader interface, which defines how to read and
            retrieve data samples.  See the core.data package for implementations, or write a custom version.
        :param output_dir: Output directory for CCP iterations.  Q-vectors and network weights will be written here.
            This directory will be created if it does not already exist.
        :param encoder_network_f_b: The encoder network to use (f_b). The encoder produces an underlying representation
            of the data produced by the data_reader. The encoder result is used for both the pseudo-labeling and
            classification tasks.
        :param projection_head_f_z: The projection head to use (f_z). The projection head operates on the output of the
            shared encoder network to produce underlying representations of data that are optimized for the
            pseudo-labeling CCP task. The output dimensionality is arbitrary. The paper uses a 2-layer MLP
            with ReLU activation function as the projection head.
        :param training_options: A class packaging the training options for CCP.  Allows for customizing optimizers and
            learning rate schedulers per "regime" (prewarming, ccp training, and classifier training).
            The `ccp_experiment_configs.py` file builds example options from the paper.
        :param transforms: A list of Transforms to draw from during contrastive learning.
        :param independently_transform_samples: Boolean flag indicating whether to randomize the transforms within a
         batch or to operate all transforms at the batch level.
        :param batch_size: The size of the batches of data to use.
        :param target_sample_rates: A dictionary mapping targets to sample rates.
          Any non-existent keys are assumed to be sampled at a rate of `1`, ie roughly balanced in a batch.
          A sample rate of 10 means take 10 samples from the target for each single pass over the targets when
          constructing a batch.  By default we upsample "unlabelled" 10x but keep all other targets at roughly balanced.
        :param random_seed: A random seed used across CCP - exposed for repeatability.
        """
        self.data_reader = data_reader
        self.dataset = CCPDataset(data_reader=self.data_reader)

        # Build a batched sampler to do custom balanced sampling - we use this batch sampler each epoch in a
        # simple DataLoader instance to generate a new random set of batches:
        self.batch_sampler = CCPBatchSampler(
            batch_size=batch_size,
            dataset_target_idxs=self.data_reader.sorted_target_idxs,
            target_sample_rates=target_sample_rates,
            random_seed=random_seed,
        )
        self.data_loader = DataLoader(
            dataset=self.dataset, batch_sampler=self.batch_sampler
        )

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.encoder = encoder_network_f_b
        self.projection_head = projection_head_f_z
        self.network_parameters = itertools.chain(
            self.encoder.parameters(), self.projection_head.parameters()
        )
        self.network_init_func = network_init_func

        self.training_options = training_options
        self.transform = TransformApply(
            transforms=transforms,
            independently_transform_samples=independently_transform_samples,
        )

        self.criterion = SoftSupervisedContrastiveLoss()

        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        self.prewarm_network_state: Optional[
            Dict
        ] = None  # Set initial pre-warmed network state

        # Reset all modules to start CCP process:
        self.initialize_parameters()

    def randomly_initialize_parameters(self):
        """
        Initializes all weights randomly.
        Most users will want `initialize_parameters`, which handles prewarmed state.
        """
        LOGGER.info("Randomly initializing network parameters!")
        for module in [self.encoder, self.projection_head]:
            module.apply(self.network_init_func)

    def initialize_parameters(self):
        """
        Initializes all weights randomly for the modules passed in 'modules'.
        """
        if self.prewarm_network_state:
            LOGGER.info("Resetting network to pre-warm state!")
            self.encoder.load_state_dict(self.prewarm_network_state["encoder"])
            self.projection_head.load_state_dict(
                self.prewarm_network_state["projection_head"]
            )
        else:
            self.randomly_initialize_parameters()

    def prewarm_network(
        self,
        exit_criteria: EMALossExitCriteria,
        print_loss_every_k_batches: int = 500,
    ):
        """
        Performs fully unsupervised contrastive pre-training on the encoder/projection head, akin to SimCLR.
        The exit condition is computed using an EMA on the loss with a threshold on the number of epochs since the last
        improvement (similar to the exit condition for any single CCP iteration).
        Model states at the end of this pretraining are persisted to self.prewarm_network_state - they are then
        preferentially used to re-initialize the network between iterations of CCP.

        To reset this behavior, users can call `reset_prewarm_network`.
        :param exit_criteria: An EMALossExitCriteria instance defining how to compute the EMA loss and when to complete
            training.
        :param print_loss_every_k_batches: How often to print the loss during an epoch of training.
        """
        LOGGER.info(
            f"Prewarming network with fully unsupervised contrastive training under regime: {self.training_options[CCPRegime.PREWARM]}"
        )
        optimizer, scheduler = self.training_options.build_training_regime(
            regime=CCPRegime.PREWARM, parameters=self.network_parameters
        )
        epochs, best_loss, _ = ema_training_loop(
            single_iteration_func=self.prewarm_network_single_epoch,
            exit_criteria=exit_criteria,
            initial_func_params={
                "optimizer": optimizer,
                "scheduler": scheduler,
                "print_loss_every_k_batches": print_loss_every_k_batches,
            },
        )

        LOGGER.info(
            f"Completed prewarming after {epochs} epochs, with a best EMA loss of {best_loss:0.6f}."
        )
        self.prewarm_network_state = {
            "encoder": copy.deepcopy(self.encoder.state_dict()),
            "projection_head": copy.deepcopy(self.projection_head.state_dict()),
        }

    def prewarm_network_single_epoch(
        self,
        optimizer: optim.Optimizer,
        scheduler: Optional[lr_scheduler.LRScheduler],
        print_loss_every_k_batches: int,
    ) -> Tuple[float, Dict]:
        """
        Prewarm encoder and projection using fully unsupervised pretraining for a single epoch.

        :param optimizer: The Optimizer to use while training over the epoch. Advances len(data_loader) steps per call.
        :param scheduler: The LRScheduler to use while training over the epoch.  Advances one step per call.
        :param print_loss_every_k_batches: How frequently to average and print the cumulative loss.
        :return: Average loss on the epoch and the default args with which another execution should be called.
        """
        cumulative_loss = 0.0
        batch_count = 0
        for b_idx, batch in enumerate(self.data_loader):
            samples, _, _, _ = batch
            samples = samples.to(DEVICE)
            # We can use the same loss function if we provide the correct q values - specifically, an <n, n/2> tensor
            # with 1s in the positions corresponding to matching transformed pairs, which simplifies CCP loss to
            # SIMCLR loss:
            q_vecs = torch.eye(samples.shape[0]).to(DEVICE)

            # Transform each batch of elements to get two transformed copies, then stack:
            batch_samples = torch.cat(
                (
                    self.transform.transform_apply(samples),
                    self.transform.transform_apply(samples),
                ),
                dim=0,
            )
            batch_qs = torch.cat((q_vecs, q_vecs), dim=0)

            # Train encoder/projection head using LSSC:
            optimizer.zero_grad()
            batch_z = self.projection_head(self.encoder(batch_samples))
            loss = self.criterion(batch_z, batch_qs)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            batch_count += 1
            # Print loss every `print_loss_every_k_batches` mini-batches
            if b_idx % print_loss_every_k_batches == (print_loss_every_k_batches - 1):
                print(f"[{b_idx + 1:5d}] loss: {cumulative_loss / batch_count:.3f}")
        if scheduler:
            scheduler.step()  # Mark end of a full epoch
        loss = cumulative_loss / batch_count
        args = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "print_loss_every_k_batches": print_loss_every_k_batches,
        }
        return loss, args

    def reset_prewarm_network(self):
        """
        Function that resets the persisted prewarmed network state.  Note that this is an irreversible operation!
        You may want to persist the contents of `self.prewarm_network_state` to disk to avoid losing model weights.
        :return:
        """
        LOGGER.warning(
            "Resetting the prewarmed network state - initializing weights for encoder and projection head "
            "will be forgotten!"
        )
        self.prewarm_network_state = None

    def prewarmed_encoder(self) -> nn.Module:
        """
        Build a copy of the encoder network f_b, with parameters initialized to prewarmed network state.
        :return: An instance of the nn.Module f_b.
        """
        if not self.prewarm_network_state:
            raise ValueError(
                "Cannot use prewarmed encoder network state, as encoder was not pretrained!"
            )
        encoder = copy.deepcopy(self.encoder)
        encoder.load_state_dict(self.prewarm_network_state["encoder"])
        return encoder

    def classification_dataset(self) -> QLabelDataset:
        """
        Pass-through method that creates a classification dataset using the current q-vector state.
        :return: A QLabelDataset.
        """
        return self.dataset.to_classification_dataset()

    def execute_ccp_single_iteration(
        self,
        exit_criteria: EMALossExitCriteria,
        previous_metadata: Optional[CCPMetadata] = None,
        output_prefix: str = "latest",
        print_loss_every_k_batches: int = 500,
    ) -> Tuple[float, Dict]:
        """
        Runs a single iteration of CCP. An "iteration" of CCP propagates the credibilities
        through multiple epochs. To properly train CCP, this method must be called repeatedly.

        See: Algorithm 1.

        :param exit_criteria: An EMALossExitCriteria instance defining how to compute the EMA loss and when to complete
            training one CCP iteration.
        :param previous_metadata: A CCPMetadata instance - can be None on the first iteration, or a set of desired
            initial CCP params.  After the first iteration, should be the result of the previous
            `execute_ccp_single_iteration` call.
        :param output_prefix: How to prefix output files for the results of this iteration (q vectors).
            Default is "latest", which will cause an overwrite on each iteration -- but can be incremented
            or changed each call to save intermediary results.
        :param print_loss_every_k_batches: How often to print the loss during an epoch of training.
        :return: Loss and dict of default args that could be used in the next iteration. The CCPMetadata
            for the single completed iteration started from calling this method appears in the args under
            the key "previous_metadata".
        """
        if previous_metadata is None:
            # Set reasonable defaults for a first iteration:
            previous_metadata = CCPMetadata(p_last=0.90, d_max=0.01)

        self.initialize_parameters()
        LOGGER.info(
            f"Training CCP under regime: {self.training_options[CCPRegime.CCP]}"
        )
        optimizer, scheduler = self.training_options.build_training_regime(
            regime=CCPRegime.CCP, parameters=self.network_parameters
        )
        epochs, best_loss, _ = ema_training_loop(
            single_iteration_func=self.execute_ccp_single_epoch,
            exit_criteria=exit_criteria,
            initial_func_params={
                "optimizer": optimizer,
                "scheduler": scheduler,
                "print_loss_every_k_batches": print_loss_every_k_batches,
            },
        )

        LOGGER.info(
            f"Completed iteration of CCP after {epochs} epochs, with a best EMA loss of {best_loss:0.6f}."
        )
        metadata = self.dataset.propagate_q_vecs(metadata=previous_metadata)
        if output_prefix:
            # Write resulting propagated q vectors:
            self.dataset.write_q_vecs(
                output_directory=self.output_dir, output_fname=f"{output_prefix}_q"
            )

        args = {
            "exit_criteria": exit_criteria,
            "previous_metadata": metadata,
            "output_prefix": output_prefix,
            "print_loss_every_k_batches": print_loss_every_k_batches,
        }
        return best_loss, args

    def execute_ccp_single_epoch(
        self,
        optimizer: optim.Optimizer,
        scheduler: Optional[lr_scheduler.LRScheduler],
        print_loss_every_k_batches: int,
    ) -> Tuple[float, Any]:
        """
        Train CCP using batches yielded by self.data_loader.
        Note that the functional batch size here is 2 * self.batch_size because every sample yields two transformed
        views.
        :param optimizer: The Optimizer to use while training over the epoch. Advances len(data_loader) steps per call.
        :param scheduler: The LRScheduler to use while training over the epoch.  Advances one step per call.
        :param print_loss_every_k_batches: How frequently to average the recent loss values
            and print them; each reported loss includes the mean loss across the most
            recent `print_loss_every_k_batches`
        :return: Average loss across the epoch and args needed for next call of this method
        """
        cumulative_loss = 0.0
        batch_count = 0
        for b_idx, batch in enumerate(self.data_loader):
            samples, q_vecs, targets, idxs = batch
            samples = samples.to(DEVICE)
            q_vecs = q_vecs.to(DEVICE)
            targets = targets.to(DEVICE)
            idxs = idxs.to(DEVICE)

            # Transform each batch of elements to get two transformed copies, then stack:
            batch_samples = torch.cat(
                (
                    self.transform.transform_apply(samples),
                    self.transform.transform_apply(samples),
                ),
                dim=0,
            )
            # Note that we also have to expand other tensors in the batch to
            # account for this effective duplication of samples:
            batch_qs = torch.cat((q_vecs, q_vecs), dim=0)
            batch_targets = torch.cat((targets, targets), dim=0)

            # Train encoder/projection head using LSSC:
            optimizer.zero_grad()
            batch_z = self.projection_head(self.encoder(batch_samples))
            loss = self.criterion(batch_z, batch_qs)
            loss.backward()
            optimizer.step()

            # Propagate batch_qs to running totals for current iteration so that
            # we can estimate the target to which each sample is most similar across
            # MANY batches rather than just a few.
            # We do not backprop into q propagation because q propagation does not affect
            # the loss.
            # We do not duplicate idxs to match the expectations of the target method.
            self.dataset.propagate_batch_q_vecs(
                batch_z.detach(),
                batch_qs.detach(),
                batch_targets.detach(),
                idxs.detach(),
            )

            cumulative_loss += loss.item()
            batch_count += 1
            # Print loss every `print_loss_every_k_batches` mini-batches
            if b_idx % print_loss_every_k_batches == (print_loss_every_k_batches - 1):
                print(f"[{b_idx + 1:5d}] loss: {cumulative_loss / batch_count:.3f}")

        if scheduler:
            scheduler.step()  # Mark end of a full epoch
        loss = cumulative_loss / batch_count
        args = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "print_loss_every_k_batches": print_loss_every_k_batches,
        }
        return loss, args

    def ema_train(
        self,
        ccp_overall_exit_criterion: EMALossExitCriteria,
        ccp_inner_loop_iteration_exit_criterion: EMALossExitCriteria,
        print_loss_every_k_batches: int,
    ) -> float:
        """
        Full training loop for learning the q-vectors. A full training loop consists of
        repeatedly calling `execute_ccp_single_iteration`, which in turn executes many
        epochs of training. We use an exponential moving average loss in both to identify
        when the loss is converging.

        :param ccp_overall_exit_criterion: An EMALossExitCriteria instance defining how
            to compute the EMA loss and when to complete the overall training.
        :param ccp_inner_loop_iteration_exit_criterion: An EMALossExitCriteria instance defining how
            to compute the EMA loss and when to complete each iteration / inner loop of training,
            which itself likely consists of many epochs.
        :param print_loss_every_k_batches: How often to print the loss during an epoch of training.
        :returns: The best loss achieved.
        """
        LOGGER.info("Training Soft-labeller.")

        num_iterations, best_loss, final_metadata = ema_training_loop(
            single_iteration_func=self.execute_ccp_single_iteration,
            exit_criteria=ccp_overall_exit_criterion,
            initial_func_params={
                "print_loss_every_k_batches": print_loss_every_k_batches,
                "exit_criteria": ccp_inner_loop_iteration_exit_criterion,
            },
        )

        LOGGER.info(
            f"Completed soft labeler training after {num_iterations} iterations, with a best EMA loss of {best_loss:0.6f}."
        )
        return best_loss
