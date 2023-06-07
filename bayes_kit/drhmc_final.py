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
        stepsize_fn: function,
        steps_fn: function,
        metric_diag: Optional[NDArray[np.float64]] = None,
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        self._model = model
        self._dim = self._model.dims()
        self._num_proposals = num_proposals
        self._stepsize_fn = stepsize_fn
        self._steps_fn = steps_fn
        self._stepsize_list = []
        self.steps_list = []
        self._metric = metric_diag or np.ones(self._dim)
        self._rng = np.random.default_rng(seed)
        self._theta = (
            init
            if (init is not None and init.shape != (0,))
            else self._rng.normal(size=self._dim)
        )

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
    
    def get_stepsize(self, **kwargs) -> float:
        stepsize = self._stepsize_fn(**kwargs)
        self._stepsize_list.append(stepsize)
        return stepsize
    
    def get_steps(self, **kwargs) -> int:
        steps = self._steps_fn(**kwargs)
        self._steps_list.append(steps) 
        return steps

    def sample(self) -> Draw:
        theta, rho = self._theta, self._rng.normal(size=self._dim)
        for k in range(self._num_proposals):
            stepsize, steps = self.get_stepsize(), self.get_steps()
            theta_prop, rho_prop = self.leapfrog(theta, rho, stepsize, steps)

            accept_logp = self.accept(theta, rho, theta_prop, rho_prop, k)
            if np.log(self._rng.uniform()) < accept_logp:
                self._theta, logp_prop = theta_prop, self.joint_logp(theta_prop, rho_prop)
                return self._theta, logp_prop
        
        logp = self.joint_logp(theta, rho)
        return self._theta, logp

    def accept(
        self, theta: NDArray[np.float64], rho: NDArray[np.float64], 
        theta_prop: NDArray[np.float64], rho_prop: NDArray[np.float64], k: int
    ):
        logp = self.joint_logp(theta, rho)
        logp_prop = self.joint_logp(theta_prop, rho_prop)
        
        # computation + recurse
        log_num = 0
        for i in range(k):
            stepsize, steps = self._stepsize_list[i], self._steps_list[i]
            theta_ghost, rho_ghost = self.leapfrog(theta_prop, rho_prop, stepsize, steps)
            accept_logp_ghost = self.accept(theta_prop, rho_prop, theta_ghost, rho_ghost, i)
            log_num += np.log(1 - np.exp(accept_logp_ghost))
            
        log_denom = 0
        for i in range(k):
            stepsize, steps = self._stepsize_list[i], self._steps_list[i]
            theta_old_prop, rho_old_prop = self.leapfrog(theta, rho, stepsize, steps)
            accept_logp_old = self.accept(theta, rho, theta_old_prop, rho_old_prop, i)
            log_denom += np.log(1 - np.exp(accept_logp_old))
        
        accept_logp = (logp_prop - logp) + (log_num - log_denom)
        return accept_logp
            
            
            
            