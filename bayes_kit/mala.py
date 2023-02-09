from typing import Iterator, Optional, Union
from numpy.typing import NDArray
import numpy as np

from .metropolis import metropolis_hastings_accept_test
from .model_types import GradModel

Sample = tuple[NDArray[np.float64], float]


class MALA:
    def __init__(
        self,
        model: GradModel,
        epsilon: float,
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        self._model = model
        self._epsilon = epsilon
        self._dim = self._model.dims()
        self._rand = np.random.default_rng(seed)
        self._theta = init or self._rand.normal(size=self._dim)
        self._log_p_theta, logpgrad = self._model.log_density_gradient(self._theta)
        self._log_p_grad_theta = np.asanyarray(logpgrad)

    def __iter__(self) -> Iterator[Sample]:
        return self

    def __next__(self) -> Sample:
        return self.sample()

    def sample(self) -> Sample:
        theta_prop = (
            self._theta
            + self._epsilon * self._log_p_grad_theta
            + np.sqrt(2 * self._epsilon) * self._rand.normal(size=self._model.dims())
        )

        lp_prop, grad_prob_al = self._model.log_density_gradient(theta_prop)
        grad_prop = np.asanyarray(grad_prob_al)

        lp_forward = self._correction(theta_prop, self._theta, self._log_p_grad_theta)
        lp_reverse = self._correction(self._theta, theta_prop, grad_prop)

        if metropolis_hastings_accept_test(
            lp_prop,
            self._log_p_theta,
            lp_forward,
            lp_reverse,
            self._rand,
        ):
            self._theta = theta_prop
            self._log_p_theta = lp_prop
            self._log_p_grad_theta = grad_prop

        return self._theta, self._log_p_theta

    def _correction(
        self,
        theta_prime: NDArray[np.float64],
        theta: NDArray[np.float64],
        grad_theta: NDArray[np.float64],
    ) -> float:
        x = theta_prime - theta - self._epsilon * grad_theta
        dot_self: float = x.dot(x)
        return (-0.25 / self._epsilon) * dot_self
