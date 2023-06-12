import numpy as np
import pydantic
from typing import Optional, Callable, NamedTuple

from bayes_kit.types import GradModel, ArrayType, SeedType
from .base_mcmc import BaseMCMC

Vector = ArrayType
ProposalFn = Callable[[Vector], Vector]
TransitionLPFn = Callable[[Vector, Vector], float]
from .metropolis import metropolis_hastings_accept_test

# Note: MALA is an instance of a Metropolis-Hastings algorithm, but we do not
# implement it here as a subclass in order to cache gradient calls.
# Naively implementing MALA as MH via callbacks requires three gradient
# calls per iteration, this implementation only requires one


class MALA(BaseMCMC):
    model: GradModel
    short_name: str = "MALA"
    description: str = "Metropolis-adjusted Langevin algorithm"

    def __init__(
        self,
        model: GradModel,
        epsilon: float,
        init: Optional[ArrayType] = None,
        seed: Optional[SeedType] = None,
    ):
        super().__init__(model=model, init=init, seed=seed)
        self._epsilon = epsilon
        self._log_p_theta, logpgrad = self.model.log_density_gradient(self._theta)
        self._log_p_grad_theta = np.asanyarray(logpgrad)

    def step(self):
        theta_prop = (
            self._theta
            + self._epsilon * self._log_p_grad_theta
            + np.sqrt(2 * self._epsilon) * self._rng.normal(size=self.model.dims())
        )
        lp_prop, grad_prop = self.model.log_density_gradient(theta_prop)
        # user-provided models can return non-NDArrays as gradients
        grad_prop = np.asanyarray(grad_prop)

        lp_forward = self._proposal_log_density(
            theta_prop, self._theta, self._log_p_grad_theta
        )
        lp_reverse = self._proposal_log_density(self._theta, theta_prop, grad_prop)

        accept = metropolis_hastings_accept_test(
            lp_prop,
            self._log_p_theta,
            lp_forward,
            lp_reverse,
            self._rng,
        )
        if accept:
            self._theta = theta_prop
            self._log_p_theta = lp_prop
            self._log_p_grad_theta = grad_prop

        return self._theta, {"logp": self._log_p_theta, "accepted": accept}

    def _proposal_log_density(
        self,
        theta_prime: ArrayType,
        theta: ArrayType,
        grad_theta: ArrayType,
    ) -> float:
        """
        Conditional log probability of the proposed parameters
        given the current parameters, log q(theta' | theta).
        """
        x = theta_prime - theta - self._epsilon * grad_theta
        return (-0.25 / self._epsilon) * x.dot(x)  # type: ignore  # dot is untyped

    class Params(BaseMCMC.Params):
        epsilon: float = pydantic.Field(description="Size of each gradient step")

        @pydantic.validator("epsilon")
        def epsilon_positive(cls, v):
            if v <= 0:
                raise ValueError("epsilon must be positive")
            return v

    @classmethod
    def new_from_params(cls, params: Params, **kwargs) -> "MALA":
        return cls(model=kwargs.pop('model'), epsilon=params.epsilon, **kwargs)

    class State(NamedTuple):
        theta: ArrayType
        rng: tuple
        epsilon: float
        log_p_theta: float
        log_p_grad_theta: ArrayType

    def get_state(self) -> State:
        return self.State(
            theta=self._theta,
            rng=self._rng.bit_generator.state,
            epsilon=self._epsilon,
            log_p_theta=self._log_p_theta,
            log_p_grad_theta=self._log_p_grad_theta,
        )

    def set_state(self, state: State):
        self._theta = state.theta
        self._rng.bit_generator.state = state.rng
        self._epsilon = state.epsilon
        self._log_p_theta = state.log_p_theta
        self._log_p_grad_theta = state.log_p_grad_theta
