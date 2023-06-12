from typing import Protocol, Any, NamedTuple, Optional, TypeVar
from .models import LogDensityModel
from .numeric import ArrayType


# StepType is a generic placeholder type that stands for "whatever the intermediate
# outputs of the inference algorithm are" from each call to step(), which are then fed
# to a DistributionApproximation.update() method.
StepType = TypeVar("StepType")
ExtrasType = dict[str, Any]


class WeightedSamples(NamedTuple):
    weights: ArrayType
    samples: ArrayType


class DistributionApproximation(Protocol[StepType]):
    def update(self, step: StepType):
        """Update the approximation state based on the outcome of an inference step.
        """
        ...

    def draw(self, n: int) -> WeightedSamples:
        """Draw n (weighted) samples from the approximation.

        If the approximation is such that it can only provide fewer than n samples, it
        should raise a ValueError.

        If the approximation is such that it does not normally provide weights, it
        should return an array of ones for the weights.

        Args:
            n: Number of samples to draw.

        Returns:
            A tuple of samples, weights, each of which is an array of length n. The
            shape of each sample depends on the underlying model.
        """
        ...


class InferenceAlgorithm(Protocol[StepType]):
    model: Optional[LogDensityModel]

    def step(self) -> tuple[StepType, ExtrasType]:
        """Take a single step of the inference algorithm.

        Returns:
            A tuple of the updated state of the inference algorithm, and any 'extra'
             information for diagnostics as a dict.
        """
        ...


class InferenceRunner(Protocol[StepType]):
    """A class that implements the InferenceRunner protocol runs a given algorithm
    to construct a given approximation. They must have compatible StepTypes.
    """
    algorithm: InferenceAlgorithm[StepType]
    approximation: DistributionApproximation[StepType]


__all__ = [
    "StepType",
    "ExtrasType",
    "WeightedSamples",
    "DistributionApproximation",
    "InferenceAlgorithm",
    "InferenceRunner",
]
