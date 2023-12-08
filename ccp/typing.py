from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from numpy.typing import NDArray

TargetLabel = int
IdxMask = TypeVar("IdxMask", bound=NDArray[numpy.int8])
TargetLabelIdxs = Dict[TargetLabel, Union[List[int], IdxMask]]


@dataclass(frozen=True)
class CCPMetadata:
    p_last: float
    d_max: float


@dataclass(frozen=True)
class TrainingOptions:
    optimizer: Type[optim.Optimizer]  # This is a type, not an instance!
    optimizer_params: Dict[str, Any]
    scheduler: Optional[
        Type[lr_scheduler.LRScheduler]
    ]  # This is a type, not an instance!
    scheduler_params: Dict[
        str, Any
    ]  # The "optimizer" param is automatically filled using above `optimizer`
