import logging
from enum import Enum
from typing import Dict, Optional, Tuple

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ccp.typing import TrainingOptions

LOGGER = logging.getLogger(__name__)


class CCPRegime(Enum):
    PREWARM = 1
    CCP = 2
    CLASSIFIER = 3


class CCPTrainingOptions(object):
    def __init__(self, regime_training_options: Dict[CCPRegime, TrainingOptions]):
        """
        Packaging class for building CCP Training Options.
        :param regime_training_options: A mapping of CCPRegimes to their respective TrainingOptions, used to construct
        optimizers/schedulers for training the various CCP components.
        """
        self.regime_training_options = {}  # Map regimes to their training options
        for regime in CCPRegime:
            if regime in regime_training_options:
                self.regime_training_options[regime] = regime_training_options[regime]
            else:
                raise ValueError(f"Regime {regime} has no training options specified!")

    def __getitem__(self, regime: CCPRegime) -> TrainingOptions:
        return self.regime_training_options[regime]

    def build_training_regime(
        self, regime: CCPRegime, parameters
    ) -> Tuple[optim.Optimizer, Optional[lr_scheduler.LRScheduler]]:
        regime_training_options = self[regime]
        optimizer = regime_training_options.optimizer(
            params=parameters, **regime_training_options.optimizer_params
        )

        if regime_training_options.scheduler:
            scheduler = regime_training_options.scheduler(
                optimizer=optimizer, **regime_training_options.scheduler_params
            )
        else:
            scheduler = None
        return optimizer, scheduler
