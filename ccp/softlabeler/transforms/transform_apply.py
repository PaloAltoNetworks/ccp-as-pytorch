from typing import Callable, List

import torch
from torchvision.transforms import Lambda


class TransformApply(object):
    def __init__(
        self, transforms: List[Callable], independently_transform_samples: bool
    ):
        """
        Class for applying a set of transforms randomly to a batch of samples.
        Will either draw a random transform from the list and apply it to the entire batch or will draw a random
        transform for each sample and apply it independently, depending on the value of independently_transform_samples.
        Note that transforms can be an arbitrarily sized list - if the transform itself is composed of multiple
        randomly chained operations this could even be a list of size 1.
        :param transforms: A list of callable transforms.  Ideally these can all operate on arbitrarily sized tensors
            (either single elements or batches) but if a transform can only operate on a batch of elements then
            it should handle raising the appropriate errors if called with an unexpectedly shaped input.
        :param independently_transform_samples:  Whether to independently draw a transform and apply it on each sample
            in the batch or to transform the entire batch in one operation.
        """
        if not len(transforms) > 0:
            raise ValueError("Must specify at least one Transform to sample from!")
        self.transforms = transforms
        self.independently_transform_samples = independently_transform_samples

    def random_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Draw a random transform from the configured transforms and apply it.
        :param x: The tensor on which to apply the transform - note that this could be a single sample or a batch.
        :return: The transformed tensor.
        """
        t_ind = torch.randint(low=0, high=len(self.transforms), size=(1,))
        return self.transforms[t_ind](x)

    def transform_apply(self, batch: torch.Tensor) -> torch.Tensor:
        return (
            Lambda(
                lambda x: torch.stack(
                    [self.random_transform(sample) for sample in batch]
                )
            )(batch)
            if self.independently_transform_samples
            else self.random_transform(batch)
        )
