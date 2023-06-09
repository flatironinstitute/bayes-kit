from typing import Iterator, Optional, Union, Tuple, Callable
from numpy.typing import NDArray
import numpy as np

from .model_types import GradModel

Draw = tuple[NDArray[np.float64], float]


class DRHMC:
    def __init__(
        self,
        model: GradModel,
        num_proposals: int,
        stepsize_fn: Callable,
        steps_fn: Callable,
        metric_diag: Optional[NDArray[np.float64]] = None,
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        self._model = model
        self._dim = self._model.dims()
        self._num_proposals = num_proposals
        self._stepsize_fn = stepsize_fn
        self._steps_fn = steps_fn
        self._metric = metric_diag or np.ones(self._dim)
        self._rng = np.random.default_rng(seed)
        self._theta = (
            init
            if (init is not None and init.shape != (0,))
            else self._rng.normal(size=self._dim)
        )
        
        self._stepsize_list = []
        self._steps_list = []

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
    
    def get_stepsize(self, *args) -> float:
        stepsize = self._stepsize_fn(*args)
        self._stepsize_list.append(stepsize)
        return stepsize
    
    def get_steps(self, *args) -> int:
        steps = self._steps_fn(*args)
        self._steps_list.append(steps) 
        return steps
    
    def sample(self) -> Draw:
        rho = self._rng.normal(size=self._dim)
        for k in range(self._num_proposals):
            stepsize, steps = self.get_stepsize(k), self.get_steps(k)
            theta_prop, rho_prop = self.leapfrog(self._theta, rho, stepsize, steps)
            rho_prop *= -1
            accept_logp = self.accept(self._theta, rho, theta_prop, rho_prop, k)
            if np.log(self._rng.uniform()) < accept_logp:
                self._theta, rho = theta_prop, rho_prop
                break
        logp = self.joint_logp(self._theta, rho)
        return self._theta, logp
    
    # def sample(self) -> Draw:
    #     rho = self._rng.normal(size=self._dim)
    #     self._theta, logp = self.sample_recurse(self._theta, rho, 0)
    #     return self._theta, logp
        
    # def sample_recurse(self, theta, rho, proposal) -> Draw:
    #     if proposal == self._num_proposals:
    #         return theta, self.joint_logp(theta, rho)
        
    #     stepsize, steps = self.get_stepsize(proposal), self.get_steps(proposal)
    #     theta_prop, rho_prop = self.leapfrog(theta, rho, stepsize, steps)
    #     rho_prop *= -1
    #     accept_logp = self.accept(theta, rho, theta_prop, rho_prop, proposal)
    #     if np.log(self._rng.uniform()) < accept_logp:
    #         return theta, self.joint_logp(theta, rho)
    #     else:
    #         return self.sample_recurse(theta, rho, proposal + 1)

    def accept(
        self, theta: NDArray[np.float64], rho: NDArray[np.float64], 
        theta_prop: NDArray[np.float64], rho_prop: NDArray[np.float64], k: int
    ) -> float:
        logp = self.joint_logp(theta, rho)
        logp_prop = self.joint_logp(theta_prop, rho_prop)
        log_num, log_denom = 0, 0
        for i in range(k):
            stepsize, steps = self._stepsize_list[i], self._steps_list[i]
            
            theta_ghost, rho_ghost = self.leapfrog(theta_prop, rho_prop, stepsize, steps)
            rho_ghost *= -1
            accept_logp = self.accept(theta_prop, rho_prop, theta_ghost, rho_ghost, i)
            reject_logp = np.log1p(-np.exp(accept_logp))
            log_num += reject_logp
            
            theta_reject, rho_reject = self.leapfrog(theta, rho, stepsize, steps)
            rho_reject *= -1
            accept_logp = self.accept(theta, rho, theta_reject, rho_reject, i)
            reject_logp = np.log1p(-np.exp(accept_logp))
            log_denom += reject_logp
        return min(0., (logp_prop - logp) + (log_num - log_denom))