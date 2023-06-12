import numpy as np
import pydantic
from bayes_kit.types import DistributionApproximation, ArrayType, WeightedSamples, PydanticNDArray


class InMemorySamplingApproximation(DistributionApproximation):
    class State(pydantic.BaseModel):
        samples: list[PydanticNDArray]
        thinning: int
        burn_in: int
        index: int

    def __init__(self, thinning: int = 1, burn_in: int = 0):
        self.samples = []
        self.thinning = thinning
        self.burn_in = burn_in
        self.index = 0

    def get_state(self) -> State:
        # Note: not trusting the caller and copying the samples list. This is likely a
        # performance bottleneck. Ideally would provide some sort of read-only view
        # or lazy copy-on-write semantics.
        return InMemorySamplingApproximation.State(
            samples=self.samples.copy(),
            thinning=self.thinning,
            burn_in=self.burn_in,
            index=self.index
        )

    def set_state(self, state: State) -> None:
        # See note on copy() in get_state()
        self.samples = state.samples.copy()
        self.thinning = state.thinning
        self.burn_in = state.burn_in
        self.index = state.index

    def update(self, step: ArrayType):
        self.index += 1
        if self.index > self.burn_in and self.index % self.thinning == 0:
            self.samples.append(step)

    def draw(self, n: int) -> WeightedSamples:
        if len(self.samples) < n:
            raise ValueError(f"Cannot draw {n} samples, only {len(self.samples)} available.")
        return WeightedSamples(weights=np.ones(n), samples=np.array(self.samples[-n:]))
