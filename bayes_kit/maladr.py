from typing import Iterator, Optional, Union, Callable
from numpy.typing import NDArray
import numpy as np

from .metropolis import metropolis_hastings_accept_test
from .model_types import GradModel

Draw = tuple[NDArray[np.float64], float]

# Note: MALA is an instance of a Metropolis-Hastings algorithm, but we do not
# implement it here as a subclass in order to cache gradient calls.
# Naively implementing MALA as MH via callbacks requires three gradient
# calls per iteration, this implementation only requires one


class MALADR:
    def __init__(
        self,
        model: GradModel,
        num_proposals: int,
        stepsize_fn: Callable,
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        self._model = model
        self._num_proposals = num_proposals
        self._stepsize_fn = stepsize_fn
        self._dim = self._model.dims()
        self._rng = np.random.default_rng(seed)
        self._theta = (
            init
            if (init is not None and init.shape != (0,))
            else self._rng.normal(size=self._dim)
        )
        self._log_p_theta, logpgrad = self._model.log_density_gradient(self._theta)
        self._log_p_grad_theta = np.asanyarray(logpgrad)
        self._stepsize_list = []

    def __iter__(self) -> Iterator[Draw]:
        return self

    def __next__(self) -> Draw:
        return self.sample()

    def get_stepsize(self, k) -> float:
        stepsize = self._stepsize_fn(k)
        self._stepsize_list.append(stepsize)
        return stepsize

    def sample(self) -> Draw:
        log_denom = 0
        for k in range(self._num_proposals):
            stepsize = self.get_stepsize(k)
            _, theta_grad = self._model.log_density_gradient(self._theta)
            theta_prop = (
                self._theta
                + stepsize * theta_grad
                + np.sqrt(2 * stepsize) * self._rng.normal(size=self._model.dims())
            )
            accept_logp = self.accept(self._theta, theta_prop, k, log_denom)
            log_denom += 1 - np.exp(accept_logp)
            if np.log(self._rng.uniform()) < accept_logp:
                self._theta = theta_prop
                break
        logp = self._model.log_density(self._theta)
        return self._theta, logp

    def accept(
        self,
        theta: NDArray[np.float64],
        theta_prop: NDArray[np.float64],
        k: int,
        log_denom: float,
    ) -> float:
        logp = self._model.log_density(theta)
        logp_prop = self._model.log_density(theta_prop)
        log_num = 0
        for i in range(k):
            stepsize = self._stepsize_list[i]
            _, theta_grad = self._model.log_density_gradient(theta_prop)
            theta_ghost = (
                theta_prop
                + stepsize * theta_grad
                + np.sqrt(2 * stepsize) * self._rng.normal(size=self._model.dims())
            )
            accept_logp = self.accept(theta_prop, theta_ghost, i, log_num)
            reject_logp = np.log1p(-np.exp(accept_logp))
            log_num += reject_logp
        logq_forward = self._proposal_log_density(theta_prop, theta, stepsize)
        logq_reverse = self._proposal_log_density(theta, theta_prop, stepsize)
        return min(
            0,
            (logp_prop - logp) + (logq_reverse - logq_forward) + (log_num - log_denom),
        )

    def _proposal_log_density(
        self,
        theta_prime: NDArray[np.float64],
        theta: NDArray[np.float64],
        stepsize: float,
    ) -> float:
        """
        Conditional log probability of the proposed parameters
        given the current parameters, log q(theta' | theta).
        """
        # user-provided models can return non-NDArrays as gradients
        _, grad = self._model.log_density_gradient(theta)
        grad = np.asanyarray(grad)
        x = theta_prime - theta - stepsize * grad
        return (-0.25 / stepsize) * x.dot(x)  # type: ignore  # dot is untyped
