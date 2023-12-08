import logging

import torch

LOGGER = logging.getLogger(__name__)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
LOGGER.info(f"Using {DEVICE} as the device")
