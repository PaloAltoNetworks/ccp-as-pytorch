import logging
from typing import Callable, NamedTuple

LOGGER = logging.getLogger(__name__)


class EMALossExitCriteria(NamedTuple):
    """
    Network training exit criteria config class that uses the exponential moving average.
    The training loop exits when exponential moving average (EMA) of the loss does not
    improve over a certain number of epochs.

    This class exposes parameters for defining a specific exit criteria (and provides reasonable defaults):
    :param ema_weight: The EMA weight to use when computing the epoch-over-epoch average loss.
    :param max_units_since_overwrite: The number of iterations/epochs to continue training without decreasing
        loss before exiting.
    :param max_total_units: The total number of iterations/epochs allowed before force-exiting.
        If max_total_units > max_units_since_overwrite, training will continue until we've spent
        max_units_since_overwrite unable to improve the loss, and training will automatically exit
        once we reach max_total_units. If max_total_units <= max_units_since_overwrite, training
        will exit after max_total_units.  To train fully, max_total_units should be a large positive
        number so we do not exit before the loss has settled at a low value. To run end-to-end quickly
        and not learn much, it may be useful to set max_total_units quite low.
    """

    ema_weight: float = 0.01
    max_units_since_overwrite: int = 10
    max_total_units: int = 1000


def ema_training_loop(
    single_iteration_func: Callable,
    exit_criteria: EMALossExitCriteria,
    initial_func_params: dict = {},
):
    """
    Helper loop for executing training until convergence using EMA exit criteria.
    Computes an EMA on the loss returned by `single_iteration_func` on each iteration and exits when sufficient units
    have passed without improvement (see exit_criteria for details).
    :param single_iteration_func:  The single-unit training function (might be an epoch or an iteration).
        Should return a tuple of the average loss over its execution and a dictionary of default arguments.
        The returned arguments will be passed to the next execution of `single_iteration_func` as-is, and
        likely should include at a minimum the keys from `initial_func_params`.
    :param initial_func_params: Dict of keyword parameters to pass to the single_iteration_func on the first call.
    :return: A tuple of the total number of epochs, the best EMA loss achieved, and the default args that
        the next iteration (which would be run outside this training loop) should use.
    """

    LOGGER.info(f"Conducting EMA training loop with exit criteria: {exit_criteria}")

    executions = 0
    executions_since_overwrite = 0
    best_loss = float("inf")
    ema_loss = None

    next_execution_args = initial_func_params
    while (executions_since_overwrite < exit_criteria.max_units_since_overwrite) and (
        executions < exit_criteria.max_total_units
    ):
        LOGGER.info(
            f"Starting epoch {executions_since_overwrite + 1} of {exit_criteria.max_units_since_overwrite}. (Total effort toward converge so far: {executions} executions.)"
        )
        # We evaluate on train loss for alignment with paper code -- would be better to eval on validation loss
        loss, next_execution_args = single_iteration_func(**next_execution_args)
        # Account for the initializing condition:
        ema_loss = (
            loss * exit_criteria.ema_weight + ema_loss * (1.0 - exit_criteria.ema_weight)  # type: ignore[operator]
            if ema_loss is not None
            else loss
        )
        if ema_loss < best_loss:
            LOGGER.info(
                f"Resetting counter (loss improved by {best_loss - ema_loss:0.6f}, from {best_loss:0.6f} to {ema_loss:0.6f})."
            )
            best_loss = ema_loss
            executions_since_overwrite = 0
        else:
            executions_since_overwrite += 1
        executions += 1
    return executions, best_loss, next_execution_args
