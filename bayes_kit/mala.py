from typing import Optional, Tuple
from numpy.typing import NDArray
import numpy as np

from .model_types import GradModel


class MALA:
    def __init__(
        self,
        model: GradModel,
        epsilon: float,
        init: Optional[NDArray[np.float64]] = None,
    ):
        self._model = model
        self._epsilon = epsilon
        self._dim = self._model.dims()
        self._theta = init or np.random.normal(size=self._dim)
        self._log_p_theta, self._log_p_grad_theta = self._model.log_density_gradient(
            self._theta
        )

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    def sample(self) -> Tuple[NDArray[np.float64], float]:
        theta_prop = (
            self._theta
            + self._epsilon * self._log_p_grad_theta
            + np.sqrt(2 * self._epsilon) * np.random.normal(size=self._model.dims())
        )

        lp_prop, grad_prop = self._model.log_density_gradient(theta_prop)

        if np.log(np.random.random()) < lp_prop + self.correction(
            self._theta, theta_prop, grad_prop
        ) - self._log_p_theta - self.correction(
            theta_prop, self._theta, self._log_p_grad_theta
        ):
            self._theta = theta_prop
            self._log_p_theta = lp_prop
            self._log_p_grad_theta = grad_prop

        return self._theta, self._log_p_theta

    def correction(self, theta_prime, theta, grad_theta):
        x = theta_prime - theta - self._epsilon * grad_theta
        return (-0.25 / self._epsilon) * x.dot(x)
