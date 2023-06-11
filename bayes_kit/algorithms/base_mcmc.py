import numpy as np
import pydantic
from abc import ABC, abstractmethod

from bayes_kit.protocols import InferenceAlgorithm, InitFromParams, ExtrasType, \
    ArrayType, SeedType
from bayes_kit.model_types import LogDensityModel
from typing import Optional, NamedTuple


class BaseMCMC(ABC, InferenceAlgorithm[ArrayType], InitFromParams):
    @property
    @abstractmethod
    def short_name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    class Params(pydantic.BaseModel):
        seed: Optional[int] = pydantic.Field(description="Random seed", default=None)

        @pydantic.validator("seed", pre=True)
        def seed_to_generator(cls, v):
            if v is None:
                return np.random.default_rng()
            elif isinstance(v, int):
                return np.random.default_rng(v)
            elif isinstance(v, (np.random.BitGenerator, np.random.Generator)):
                return v
            else:
                raise ValueError("seed must be None, int, or np.random.Generator")

    @classmethod
    @abstractmethod
    def new_from_params(cls, params: Params, **kwargs) -> "BaseMCMC":
        ...

    # Ensure that subclasses implement the HasState Protocol

    class State(NamedTuple):
        ...

    @abstractmethod
    def get_state(self) -> State:
        ...

    @abstractmethod
    def set_state(self, state: State):
        ...

    # Base init handles rng and init size checking

    def __init__(
            self,
            model: LogDensityModel,
            init: Optional[ArrayType] = None,
            seed: SeedType = None):
        self.model = model
        self._dim = self.model.dims()
        self._rng = np.random.default_rng(seed)
        self._theta = (
            init
            if (init is not None and init.shape != (0,))
            else self._rng.normal(size=self._dim)
        )

    @abstractmethod
    def step(self) -> tuple[ArrayType, ExtrasType]:
        ...

    # Make samplers iterable
    def __iter__(self):
        return self

    def __next__(self):
        return self.step()
