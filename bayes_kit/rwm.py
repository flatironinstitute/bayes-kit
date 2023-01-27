from typing import Callable, Iterator, Optional, Union
from numpy.typing import NDArray, ArrayLike
import numpy as np

from .model_types import LogDensityModel

Sample = tuple[NDArray[np.float64], float]


class RandomWalkMetropolis:
    def __init__(
        self,
        model: LogDensityModel,
        proposal_rng: Callable[[NDArray[np.float64]], ArrayLike],
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        self._model = model
        self._dim = self._model.dims()
        self._rand = np.random.default_rng(seed)
        self._proposal_rng = proposal_rng
        self._theta = init or self._rand.normal(size=self._dim)
        self._log_p_theta = self._model.log_density(self._theta)

    def __iter__(self) -> Iterator[Sample]:
        return self

    def __next__(self) -> Sample:
        return self.sample()

    def sample(self) -> Sample:
        # does not include initial value as first draw
        theta_star = np.asanyarray(self._proposal_rng(self._theta))
        log_p_theta_star = self._model.log_density(theta_star)
        if np.log(self._rand.uniform()) < log_p_theta_star - self._log_p_theta:
            self._theta = np.asanyarray(theta_star)
            self._log_p_theta = log_p_theta_star
        return self._theta, self._log_p_theta
    
