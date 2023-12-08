import torch.nn as nn


def init_weights_ccp(m: nn.Module):
    """
    Default CCP Paper parameter initialization:
    * Weight uses a truncated normal with stdev 0.1, and truncation at 2 standard
      deviations from the mean of 0
    * Biases uses a constant 0.1
    """
    if isinstance(m, nn.Module) and hasattr(m, "weight"):
        target_std = 0.1
        nn.init.trunc_normal_(
            m.weight, std=target_std, a=-2 * target_std, b=2 * target_std  # type: ignore
        )
    if isinstance(m, nn.Module) and hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias, 0.1)  # type: ignore
