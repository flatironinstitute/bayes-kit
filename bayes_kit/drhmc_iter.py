from typing import Iterator, Optional, Union
from numpy.typing import NDArray
import numpy as np

from .model_types import GradModel

Draw = tuple[NDArray[np.float64], float]


class DRHMC:
    def __init__(
        self,
        model: GradModel,
        stepsize: function,
        steps: function,
        num_proposals: int,
        metric_diag: Optional[NDArray[np.float64]] = None,
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        self._model = model
        self._dim = self._model.dims()
        self._stepsize = stepsize
        self._steps = steps
        self._num_proposals = num_proposals
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

    # position: theta == q
    # momentum: rho == p
    # where do they negate momentum p?
    # change self._stepsize to function that computes stepsize, or float, or list.
    
    def get_stepsize(self) -> float:
        self._stepsizes.append(self._stepsize) 
        return self._stepsizes[-1]
    
    def get_steps(self) -> int:
        
    
    def ghost(
        self, theta_prop: NDArray[np.float64], rho_prop: NDArray[np.float64], k: int
    ) -> float:
        logp = self.joint_logp(theta_prop, rho_prop)
        
        prev, log_denom = -np.inf, 0
        for i in range(k):
            stepsize = self._stepsizes[i]
            theta_ghost, rho_ghost = self.leapfrog(theta_prop, rho_prop, stepsize)
            logp_ghost = self.joint_logp(theta_ghost, rho_ghost)
            
            log_denom += np.log(1 - np.exp(prev))
            log_num = self.ghost(theta_prop, rho_prop, k)
            
            prev = accept_logp
            accept_logp = (logp_ghost - logp) + (log_num - log_denom)

        return accept_logp
    
    
    def sample(self) -> Draw:
        rho = self._rng.normal(size=self._dim)
        logp = self.joint_logp(self._theta, rho)
        
        mem_logp = dict() # mem[(theta, rho)] = logp
        mem_accept = dict() # mem[(theta, rho, proposal)] = accept_logp
        mem_logp[(self._theta, rho)] = logp
        
        for k in range(self._num_proposals):
            stepsize, steps = self.get_stepsize(), self.get_steps()
            theta_prop, rho_prop = self.leapfrog(self._theta, rho, stepsize, steps)
            logp_prop = self.joint_logp(theta_prop, rho_prop)
            
            for i in range(k-1):
                stepsize, steps = self._stepsizes[i], self._steps[i]
                theta_prop, rho_prop = self.leapfrog(self._theta, rho, stepsize, steps)
                
                if (theta_prop, rho_prop) in accept_logp:
                    accept_logp = mem_accept[(theta_prop, rho_prop, i)]
                else:
                        
                
            
            accept_logp = (logp_prop - logp)
            
            # mem_logp[(theta_prop, rho_prop)] = logp_prop
            # mem_accept[(theta_prop, rho_prop, k)] = accept_logp
            
            if np.log(self._rng.uniform()) < accept_logp:
                self._theta = theta_prop
                return self._theta, logp_prop
        return self._theta, logp
            
    
    def sample(self) -> Draw:
        rho = self._rng.normal(size=self._dim)
        logp = self.joint_logp(self._theta, rho)
        
        prev, log_denom = -np.inf, 0
        for k in range(self._num_proposals):    
            stepsize, steps = self.get_stepsize(), self.get_steps()
            theta_prop, rho_prop = self.leapfrog(self._theta, rho, stepsize, steps)
            logp_prop = self.joint_logp(theta_prop, rho_prop)
            
            log_denom += np.log(1 - np.exp(prev))
            log_num = self.ghost(theta_prop, rho_prop, k)
            
            prev = accept_logp
            accept_logp = (logp_prop - logp) + (log_num - log_denom)
            
            if np.log(self._rng.uniform()) < accept_logp:
                self._theta = theta_prop
                return self._theta, logp_prop
        return self._theta, logp
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    