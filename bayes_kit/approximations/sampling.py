import numpy as np
from typing import NamedTuple

from bayes_kit.protocols import DistributionApproximation, ArrayType, WeightedSamples


class InMemorySamplingApproximation(DistributionApproximation):
    class State(NamedTuple):
        samples: list[ArrayType]
        thinning: int
        index: int

    def __init__(self, thinning: int = 1):
        self.samples = []
        self.thinning = thinning
        self.index = 0

    def get_state(self) -> State:
        return InMemorySamplingApproximation.State(
            samples=self.samples.copy(),
            thinning=self.thinning,
            index=self.index
        )

    def set_state(self, state: State) -> None:
        self.samples = state.samples.copy()
        self.thinning = state.thinning
        self.index = state.index

    def update(self, step: ArrayType):
        self.index += 1
        if self.index % self.thinning == 1:
            self.samples.append(step)

    def draw(self, n: int) -> WeightedSamples:
        if len(self.samples) < n:
            raise ValueError(f"Cannot draw {n} samples, only {len(self.samples)} available.")
        return WeightedSamples(weights=np.ones(n), samples=np.array(self.samples[-n:]))
