from typing import Callable, Optional, Tuple
from numpy.typing import NDArray, ArrayLike
import numpy as np

from .model_types import LogDensityModel

class RandomWalkMetropolis:
    def __init__(
        self,
        model: LogDensityModel,
        proposal_rng: Callable[[NDArray[np.float64]], ArrayLike],
        init: Optional[NDArray[np.float64]] = None,
    ):
        self._model = model
        self._dim = self._model.dims()
        self._proposal_rng = proposal_rng
        self._theta = init or np.random.normal(size=self._dim)
        self._log_p_theta = self._model.log_density(self._theta)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    def sample(self) -> Tuple[NDArray[np.float64], float]:
        # does not include initial value as first draw
        theta_star = self._proposal_rng(self._theta)
        log_p_theta_star = self._model.log_density(theta_star)
        if np.log(np.random.uniform()) < log_p_theta_star - self._log_p_theta:
            self._theta = np.asanyarray(theta_star)
            self._log_p_theta = log_p_theta_star
        return self._theta, self._log_p_theta


