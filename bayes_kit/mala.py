from typing import Iterator, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .metropolis import metropolis_hastings_accept_test
from .model_types import GradModel

Draw = tuple[NDArray[np.float64], float]

# Note: MALA is an instance of a Metropolis-Hastings algorithm, but we do not
# implement it here as a subclass in order to cache gradient calls.
# Naively implementing MALA as MH via callbacks requires three gradient
# calls per iteration, this implementation only requires one


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
        self._rng = np.random.default_rng(seed)
        self._theta = (
            init
            if (init is not None and init.shape != (0,))
            else self._rng.normal(size=self._dim)
        )
        self._log_p_theta, logpgrad = self._model.log_density_gradient(self._theta)
        self._log_p_grad_theta = np.asanyarray(logpgrad)

    def __iter__(self) -> Iterator[Draw]:
        return self

    def __next__(self) -> Draw:
        return self.sample()

    def sample(self) -> Draw:
        theta_prop = (
            self._theta
            + self._epsilon * self._log_p_grad_theta
            + np.sqrt(2 * self._epsilon) * self._rng.normal(size=self._model.dims())
        )
        lp_prop, grad_prop = self._model.log_density_gradient(theta_prop)
        # user-provided models can return non-NDArrays as gradients
        grad_prop = np.asanyarray(grad_prop)

        lp_forward = self._proposal_log_density(
            theta_prop, self._theta, self._log_p_grad_theta
        )
        lp_reverse = self._proposal_log_density(self._theta, theta_prop, grad_prop)

        if metropolis_hastings_accept_test(
            lp_prop,
            self._log_p_theta,
            lp_forward,
            lp_reverse,
            self._rng,
        ):
            self._theta = theta_prop
            self._log_p_theta = lp_prop
            self._log_p_grad_theta = grad_prop

        return self._theta, self._log_p_theta

    def _proposal_log_density(
        self,
        theta_prime: NDArray[np.float64],
        theta: NDArray[np.float64],
        grad_theta: NDArray[np.float64],
    ) -> float:
        """
        Conditional log probability of the proposed parameters
        given the current parameters, log q(theta' | theta).
        """
        x = theta_prime - theta - self._epsilon * grad_theta
        return (-0.25 / self._epsilon) * x.dot(x)  # type: ignore  # dot is untyped
