from types import LogDensityModel, VectorType, FloatType
from typing import Callable, Optional, Tuple
from numpy.typing import NDArray
import numpy as np

class Metropolis(Sampler):
    def __init__(
        self,
        model: LogDensityModel,
        proposal_rng: Callable[[VectorType], VectorType],
        init: Optional[VectorType] = None,
    ):
        self._model = model
        self._proposal_rng = proposal_rng
        self._theta = init or np.random.normal(size=self._model.dims())
        self._log_p_theta = self._model.log_density(self._theta)

    def __iter__(self) -> Iterator[VectorType]:
        return self

    def __next__(self) -> VectorType:
        return self.sample()

    def sample(self) -> VectorType
        # does not include initial value as first draw
        theta_star = self._proposal_rng(self._theta)
        log_p_theta_star = self._model.log_density(theta_star)
        if np.log(np.random.uniform()) < log_p_theta_star - self._log_p_theta:
            self._theta = theta_star
            self._log_p_theta = log_p_theta_star
        return self._theta
