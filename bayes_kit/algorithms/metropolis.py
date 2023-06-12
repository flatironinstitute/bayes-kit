import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable
from copy import deepcopy
import pydantic

from bayes_kit.types import LogDensityModel, ArrayType, SeedType
from .base_mcmc import BaseMCMC

ProposalFn = Callable[[ArrayType, np.random.Generator], ArrayLike]
TransitionLPFn = Callable[[ArrayType, ArrayType], float]


def metropolis_accept_test(
    lp_proposal: float, lp_current: float, rng: np.random.Generator
) -> bool:
    """
    Return whether to accept a proposed new state.

    The Metropolis acceptance condition compares the likelihood of the proposal with the likelihood
    of the current value, as a ratio L(proposal)/L(current). The proposal is accepted if this ratio
    exceeds a uniformly-random draw from [0, 1) (so, acceptance is proportional to the likelihood ratio).

    Parameters:
        lp_proposal (float): log probability of the proposed parameter values (theta-star)
        lp_current (float): log probability of the current parameter values (theta)
        rng (np.random.Generator): an appropriately seeded pseudorandom number generator

    Returns:
        True if the proposal is accepted; false otherwise
    """
    # Since we already have log likelihoods, use:
    #       log(x/y) = log(x) - log(y)
    # and test
    #       log(uniform_random) < log(x) - log(y)
    # instead of
    #       uniform_random < x/y
    log_acceptance_ratio = lp_proposal - lp_current
    log_uniform_random: float = np.log(rng.uniform())
    return log_uniform_random < log_acceptance_ratio


def metropolis_hastings_accept_test(
    lp_proposal: float,
    lp_current: float,
    lp_forward_transition: float,
    lp_reverse_transition: float,
    rng: np.random.Generator,
) -> bool:
    """Return whether to accept a proposed new state, given an asymmetric proposal distribution.

    The Metropolis-Hastings acceptance condition modifies the Metropolis condition to accommodate
    asymmetric proposal functions (i.e. where the probability of proposing theta-star, when starting from
    theta, is not equal to the probability of proposing theta when starting from theta-star.)
    This is achieved by reweighting the likelihood ratio of the states:
        Pr(proposal) / Pr(current)
    by the relative likelihood of transitioning between them:
        Pr(proposal | current) / Pr(current | proposal)
    so that the overall acceptance ratio is Pr(p)/Pr(c) * Pr(p|c)/Pr(c|p).

    Args:
        lp_proposal (float): log probability of the proposed parameter values (theta-star)
        lp_current (float): log probability of the current parameter values (theta)
        lp_forward_transition (float): log probability of proposing the proposal starting from current
        lp_reverse_transition (float): log probability of proposing the current state starting from the proposal
        rng (np.random.Generator): an appropriately seeded pseudorandom number generator

    Returns:
        True if the proposal is accepted; false otherwise
    """
    # Since we have log likelihoods, replace multiplication/division with addition/subtraction.
    log_acceptance_ratio = lp_proposal - lp_current
    log_transition_likelihood_ratio = lp_reverse_transition - lp_forward_transition
    adjusted_log_acceptance_ratio = (
        log_acceptance_ratio + log_transition_likelihood_ratio
    )
    log_uniform_random: float = np.log(rng.uniform())
    return log_uniform_random < adjusted_log_acceptance_ratio


class MetropolisHastings(BaseMCMC):

    model: LogDensityModel  # Declare the model subtype to make type checkers happy
    short_name = "MetropolisHastings"
    description = "Metropolis Hastings sampler"

    def __init__(
        self,
        model: LogDensityModel,
        proposal_fn: ProposalFn,
        transition_lp_fn: TransitionLPFn,
        *,
        init: Optional[ArrayType] = None,
        seed: Optional[SeedType] = None,
        prop_seed: Optional[SeedType] = None,
    ):
        super().__init__(model=model, init=init, seed=seed)
        self._proposal_fn = proposal_fn
        self._transition_lp_fn = transition_lp_fn
        self._log_p_theta = self.model.log_density(self._theta)
        # Semantically, if seed is set but prop_seed is not, expected behavior is that
        # everything should nonetheless be reproducible from seed. To handle this case,
        # we'll match the prop seed to the main seed but then advance the prop generator
        # by a bit so that they don't match value-for-value.
        if seed is not None and prop_seed is None:
            self._prop_rng = deepcopy(self._rng)
            self._prop_rng.bytes(128)
        else:
            self._prop_rng = np.random.default_rng(prop_seed)

    def step(self):
        proposal, lp_proposal = self._propose()
        accepted = self._accept_test(lp_proposal, proposal)
        if accepted:
            self._update_theta(proposal, lp_proposal)
        return self._theta, {"logp": self._log_p_theta, "accepted": accepted}

    def _propose(self) -> tuple[ArrayType, float]:
        untyped_proposed_theta = np.asanyarray(
            self._proposal_fn(self._theta, self._rng), dtype=np.float64
        )
        proposed_theta: ArrayType = untyped_proposed_theta
        lp_proposed_theta = self.model.log_density(proposed_theta)
        return (proposed_theta, lp_proposed_theta)

    def _update_theta(self, theta: ArrayType, log_p_theta: float) -> None:
        self._theta = theta
        self._log_p_theta = log_p_theta

    def _accept_test(self, lp_proposal: float, proposal: ArrayType) -> bool:
        lp_forward_transition = self._transition_lp_fn(proposal, self._theta)
        lp_reverse_transition = self._transition_lp_fn(self._theta, proposal)
        return metropolis_hastings_accept_test(
            lp_proposal,
            self._log_p_theta,
            lp_forward_transition,
            lp_reverse_transition,
            self._rng,
        )

    class Params(BaseMCMC.Params):
        # TODO: let proposal_fn and transition_lp_fn be specified by command-line
        #  arguments by implementing a proposal distribution factory
        prop_seed: Optional[int] = pydantic.Field(description="Random seed for proposal function", default=None)

        @pydantic.validator("prop_seed", pre=True)
        def seed_to_generator(cls, v):
            if v is None:
                return np.random.default_rng()
            elif isinstance(v, int):
                return np.random.default_rng(v)
            elif isinstance(v, (np.random.BitGenerator, np.random.Generator)):
                return v
            else:
                raise ValueError("seed must be None, int, or np.random.Generator")

    class State(BaseMCMC.State):
        prop_rng: tuple | dict
        logp: float
        # TODO - instead of storing names, store the functions themselves
        prop_fn_name: str
        transition_lp_fn_name: str

    def get_state(self) -> pydantic.BaseModel:
        return MetropolisHastings.State(
            prop_rng=self._prop_rng.bit_generator.state,
            logp=self._log_p_theta,
            prop_fn_name=self._proposal_fn.__name__,
            transition_lp_fn_name=self._transition_lp_fn.__name__,
            **super().get_state().dict(),
        )

    def set_state(self, state: pydantic.BaseModel):
        state = MetropolisHastings.State(**state.dict())
        super().set_state(state)
        self._prop_rng.bit_generator.state = state.prop_rng
        self._log_p_theta = state.logp
        assert self._proposal_fn.__name__ == state.prop_fn_name, \
            "Mismatch in proposal_fn name between self and state"
        assert self._transition_lp_fn.__name__ == state.transition_lp_fn_name, \
            "Mismatch in transition_lp_fn name between self and state"


class Metropolis(MetropolisHastings):
    def __init__(
        self,
        model: LogDensityModel,
        proposal_fn: ProposalFn,
        *,
        init: Optional[ArrayType] = None,
        seed: Optional[SeedType] = None,
        prop_seed: Optional[SeedType] = None,
    ):
        # This transition function will never be used--it isn't needed for Metropolis,
        # for which the transition probabilities are symmetric. But we need a valid one
        # for the superclass' constructor.
        dummy_transition_fn: TransitionLPFn = lambda a, b: 1
        super().__init__(model,
                         proposal_fn=proposal_fn,
                         transition_lp_fn=dummy_transition_fn,
                         init=init,
                         seed=seed,
                         prop_seed=prop_seed)

    # 'proposal' isn't used, but we need signature consistency to override the parent method
    def _accept_test(self, lp_proposal: float, proposal: ArrayType) -> bool:
        return metropolis_accept_test(lp_proposal, self._log_p_theta, self._rng)
