from typing import Callable, Optional, Tuple
from numpy.typing import NDArray
import numpy as np

from model_types import GradModel

class HMCDiag:
    def __init__(
        self,
        model: GradModel,
        stepsize: float,
        steps: int,
        metric_diag: Optional[NDArray[np.float64]] = None,
        init: Optional[NDArray[np.float64]] = None,
    ):
        self._model = model
        self._dim = self._model.dims()
        self._stepsize = stepsize
        self._steps = steps
        self._metric = metric_diag or np.ones(self._dim)
        self._theta = init or np.random.normal(size=self._dim)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    def joint_logp(self, theta: NDArray[np.float64], rho: NDArray[np.float64]) -> float:
        return self._model.log_density(theta) - 0.5 * np.dot(
            rho, np.multiply(self._metric, rho)
        )

    def leapfrog(
        self, theta: NDArray[np.float64], rho: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        # TODO(bob-carpenter): refactor to share non-initial and non-final updates
        for n in range(self._steps):
            lp, grad = self._model.log_density_gradient(theta)
            rho_mid = rho + 0.5 * self._stepsize * np.multiply(self._metric, grad)
            theta = theta + self._stepsize * rho_mid
            lp, grad = self._model.log_density_gradient(theta)
            rho = rho_mid + 0.5 * self._stepsize * np.multiply(self._metric, grad)
        return (theta, rho)

    def sample(self) -> Tuple[NDArray[np.float64], float]:
        rho = np.random.normal(size=self._dim)
        logp = self.joint_logp(self._theta, rho)
        theta_prop, rho_prop = self.leapfrog(self._theta, rho)
        logp_prop = self.joint_logp(theta_prop, rho_prop)
        if np.log(np.random.uniform()) < logp_prop - logp:
            self._theta = theta_prop
            return self._theta, logp_prop
        return self._theta, logp
