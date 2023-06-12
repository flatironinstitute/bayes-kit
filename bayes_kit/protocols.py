import numpy as np
import pydantic
from bayes_kit.model_types import LogDensityModel
from typing import Protocol, Any, NamedTuple, Optional, TypeVar, Union
from numpy.typing import NDArray

ArrayType = NDArray[np.float64]
SeedType = Union[int, np.random.BitGenerator, np.random.Generator]

# StepType is a generic placeholder type that stands for "whatever the intermediate
# outputs of the inference algorithm are" from each call to step(), which are then fed
# to a DistributionApproximation.update() method.
StepType = TypeVar("StepType")
ExtrasType = dict[str, Any]


class WeightedSamples(NamedTuple):
    weights: ArrayType
    samples: ArrayType


class HasState(Protocol):
    """A class that implements the HasState protocol must be able to serialize its
    state into a tuple (or namedtuple) using get_state and deserialize using set_state
    """

    def get_state(self) -> tuple:
        """Get a copy of the current state or parameters.
        """
        ...

    def set_state(self, state: tuple) -> None:
        """Set the state or parameters.
        """
        ...


class DistributionApproximation(HasState, Protocol[StepType]):
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


class InferenceAlgorithm(HasState, Protocol[StepType]):
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


class InitFromParams(Protocol):
    """A class that implements the InitFromParams protocol can be easily configured from
    the command line using the pydantic_cli package.

    All subclasses of InitFromParams must define a nested class called Params of type
    pydantic.BaseModel, and all inner Params classes should follow good pydantic style
    such as implementing validators.

    Command line arguments are automatically constructed from the inner Params class.

    Example:
        class MyInitFromParams(InitFromParams):
            class Params(pydantic.BaseModel):
                my_field: int = pydantic.Field(..., cli=('-f', '--my-field'))
                my_other_field : str = pydantic.Field(..., default='foo')
    """

    # It is better to have the Protocol declare a @property than an attribute here
    # because this lets us satisfy the Protocol with an ABC that declares the property
    # as abstract. This protocol is still satisfied by classes with simple short_name
    # and description class attributes. See https://stackoverflow.com/a/68339603/1935085
    @property
    def short_name(self) -> str:
        return ""

    @property
    def description(self) -> str:
        return ""

    class Params(pydantic.BaseModel):
        """Inner class defining parameters available at initialization time.
        """
        pass

    # Note that Protocols cannot specify an __init__, so instead we specify a factory
    # method that creates an instance of the class from params. This has the added
    # benefit that classes can design their own __init__ methods without worrying about
    # Params objects.

    @classmethod
    def new_from_params(cls, params: Params, **kwargs) -> "InitFromParams":
        ...
