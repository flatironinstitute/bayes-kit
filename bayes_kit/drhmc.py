from typing import Iterator, Optional, Union
from numpy.typing import NDArray
import numpy as np

from .model_types import GradModel

Draw = tuple[NDArray[np.float64], float]


class DRHMC:
    def __init__(
        self,
        model: GradModel,
        num_proposals: int,
        stepsize: Union[float, NDArray[np.float64], function] = None,
        steps: Union[float, NDArray[np.float64], function] = None,
        metric_diag: Optional[NDArray[np.float64]] = None,
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        self._model = model
        self._dim = self._model.dims()
        self._num_proposals = num_proposals
        self._stepsize = stepsize # TODO: add function
        self._steps = steps # TODO: add function
        self._metric = metric_diag or np.ones(self._dim)
        self._rng = np.random.default_rng(seed)
        self._theta = (
            init
            if (init is not None and init.shape != (0,))
            else self._rng.normal(size=self._dim)
        )
        self._rho = self._rng.normal(size=self._dim)

    def __iter__(self) -> Iterator[Draw]:
        return self

    def __next__(self) -> Draw:
        return self.sample()

    def joint_logp(self, theta: NDArray[np.float64], rho: NDArray[np.float64]) -> float:
        adj: float = -0.5 * np.dot(rho, self._metric * rho)
        return self._model.log_density(theta) + adj 

    def leapfrog(
        self, theta: NDArray[np.float64], rho: NDArray[np.float64], stepsize: float, 
        steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lp, grad = self._model.log_density_gradient(theta)
        rho_mid = rho - 0.5 * stepsize * np.multiply(self._metric, grad)
        for n in range(steps):
            rho_mid += stepsize * np.multiply(self._metric, grad)
            theta += stepsize * rho_mid
            lp, grad = self._model.log_density_gradient(theta)
        rho = rho_mid + 0.5 * stepsize * np.multiply(self._metric, grad)
        return (theta, rho)
    
    def get_stepsize(self) -> float:
        self._stepsizes.append(self._stepsize) 
        return self._stepsizes[-1]
    
    def get_steps(self) -> int:
        self._steps.append(self._steps) 
        return self._steps[-1]
    
    def sample(self) -> Draw:
        logp = self.joint_logp(self._theta, self._rho)
        if self._num_proposals == 0:
            return self._theta, logp
        self._num_proposals -= 1
        
        stepsize, steps = self.get_stepsize(), self.get_steps()
        theta_prop, rho_prop = self.leapfrog(self._theta, self._rho, stepsize, steps)
        logp_prop = self.joint_logp(theta_prop, rho_prop)
        
        
        
        if np.log(self._rng.uniform()) < logp_prop - logp:
            return theta_prop, logp_prop
        else:
            return self._theta, logp
