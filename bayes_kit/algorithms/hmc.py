import numpy as np
import pydantic
from pydantic_numpy import NDArray as PydanticNDArray
from typing import Optional, NamedTuple

from bayes_kit.types import GradModel, ArrayType, SeedType
from .base_mcmc import BaseMCMC


class HMCDiag(BaseMCMC):
    model: GradModel  # Declare the model subtype to make type checkers happy
    short_name = "HMCDiag"
    description = "Hamiltonian Monte Carlo with diagonal metric"

    def __init__(
        self,
        model: GradModel,
        stepsize: float,
        steps: int,
        metric_diag: Optional[ArrayType] = None,
        *,
        init: Optional[ArrayType] = None,
        seed: Optional[SeedType] = None,
    ):
        super().__init__(model=model, init=init, seed=seed)
        self._stepsize = stepsize
        self._steps = steps
        self._metric = metric_diag or np.ones(self._dim)

    def joint_logp(self, theta: ArrayType, rho: ArrayType) -> float:
        adj: float = 0.5 * np.dot(rho, self._metric * rho)
        return self.model.log_density(theta) - adj

    def leapfrog(
        self, theta: ArrayType, rho: ArrayType
    ) -> tuple[ArrayType, ArrayType]:
        # Initialize rho_mid by going backwards half a step so that the first full-step inside the loop brings rho_mid
        # up to +1/2 steps. Note that if self._steps == 0, the loop is skipped and the -0.5 and +0.5 steps cancel.
        lp, grad = self.model.log_density_gradient(theta)
        rho_mid = rho - 0.5 * self._stepsize * np.multiply(self._metric, grad)
        for n in range(self._steps):
            rho_mid = rho_mid + self._stepsize * np.multiply(self._metric, grad)
            theta = theta + self._stepsize * rho_mid
            lp, grad = self.model.log_density_gradient(theta)
        # Final half-step for rho
        rho = rho_mid + 0.5 * self._stepsize * np.multiply(self._metric, grad)
        return (theta, rho)

    def step(self):
        rho = self._rng.normal(size=self._dim)
        logp = self.joint_logp(self._theta, rho)
        theta_prop, rho_prop = self.leapfrog(self._theta, rho)
        logp_prop = self.joint_logp(theta_prop, rho_prop)
        if np.log(self._rng.uniform()) < logp_prop - logp:
            self._theta = theta_prop
            return self._theta, {"logp": logp_prop, "accepted": True}
        return self._theta, {"logp": logp, "accepted": False}

    class Params(BaseMCMC.Params):
        stepsize: float = pydantic.Field(description="Size of each leapfrog step")
        steps: int = pydantic.Field(description="Number of leapfrog steps")
        metric_diag: Optional[PydanticNDArray] = pydantic.Field(description="Diagonal of metric matrix", default=None)

        @pydantic.validator("stepsize")
        def stepsize_positive(cls, v):
            if v <= 0:
                raise ValueError("stepsize must be positive")
            return v

        @pydantic.validator("steps")
        def steps_positive(cls, v):
            if v <= 0:
                raise ValueError("steps must be positive")
            return v

        @pydantic.validator("metric_diag")
        def metric_diag_positive(cls, v):
            if np.any(v <= 0):
                raise ValueError("metric_diag must be positive")
            return v

    @classmethod
    def new_from_params(cls, params: Params, **kwargs) -> "HMCDiag":
        return cls(model=kwargs.pop('model'),
                   stepsize=params.stepsize,
                   steps=params.steps,
                   metric_diag=params.metric_diag,
                   seed=params.seed)

    class State(BaseMCMC.State):
        stepsize: float
        steps: int
        metric: PydanticNDArray

    def get_state(self) -> pydantic.BaseModel:
        return HMCDiag.State(
            stepsize=self._stepsize,
            steps=self._steps,
            metric=self._metric,
            **super().get_state().dict())

    def set_state(self, state: pydantic.BaseModel):
        state = HMCDiag.State(**state.dict())
        super().set_state(state)
        self._stepsize = state.stepsize
        self._steps = state.steps
        self._metric = state.metric
