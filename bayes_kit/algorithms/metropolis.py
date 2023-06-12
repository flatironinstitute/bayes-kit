import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable, NamedTuple

from bayes_kit.protocols import ArrayType, SeedType
from bayes_kit.model_types import LogDensityModel
from .base_mcmc import BaseMCMC

ProposalFn = Callable[[ArrayType], ArrayLike]
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
    ):
        super().__init__(model=model, init=init, seed=seed)
        self._proposal_fn = proposal_fn
        self._transition_lp_fn = transition_lp_fn
        self._log_p_theta = self.model.log_density(self._theta)

    def step(self):
        proposal, lp_proposal = self._propose()
        accepted = self._accept_test(lp_proposal, proposal)
        if accepted:
            self._update_theta(proposal, lp_proposal)
        return self._theta, {"logp": self._log_p_theta, "accepted": accepted}

    def _propose(self) -> tuple[ArrayType, float]:
        untyped_proposed_theta = np.asanyarray(
            self._proposal_fn(self._theta), dtype=np.float64
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
        ...

    class State(NamedTuple):
        theta: ArrayType
        rng: tuple
        logp: float
        # TODO - instead of storing names, store the functions themselves
        prop_fn_name: str
        transition_lp_fn_name: str

    def get_state(self) -> State:
        return MetropolisHastings.State(theta=self._theta,
                                        rng=self._rng.bit_generator.state,
                                        logp=self._log_p_theta,
                                        prop_fn_name=self._proposal_fn.__name__,
                                        transition_lp_fn_name=self._transition_lp_fn.__name__)

    def set_state(self, state: State):
        self._theta = state.theta
        self._rng.bit_generator.state = state.rng
        self._log_p_theta = state.logp
        assert self._proposal_fn.__name__ == state.prop_fn_name, \
            "Mismatch in proposal_fn name between self and state"
        assert self._transition_lp_fn.__name__ == state.transition_lp_fn_name, \
            "Mismatch in transition_lp_fn name between self and state"

    @classmethod
    def new_from_params(cls, params: BaseMCMC.Params, **kwargs) -> "MetropolisHastings":
        if "proposal_fn" not in kwargs:
            raise ValueError("proposal_fn must be specified")
        if "transition_lp_fn" not in kwargs:
            raise ValueError("transition_lp_fn must be specified")
        return cls(model=kwargs.pop('model'),
                   proposal_fn=kwargs.pop('proposal_fn'),
                   transition_lp_fn=kwargs.pop('transition_lp_fn'),
                   seed=params.seed)


class Metropolis(MetropolisHastings):
    def __init__(
        self,
        model: LogDensityModel,
        proposal_fn: ProposalFn,
        *,
        init: Optional[ArrayType] = None,
        seed: Optional[SeedType] = None,
    ):
        # This transition function will never be used--it isn't needed for Metropolis,
        # for which the transition probabilities are symmetric. But we need a valid one
        # for the superclass' constructor.
        dummy_transition_fn: TransitionLPFn = lambda a, b: 1
        super().__init__(model,
                         proposal_fn=proposal_fn,
                         transition_lp_fn=dummy_transition_fn,
                         init=init,
                         seed=seed)

    # 'proposal' isn't used, but we need signature consistency to override the parent method
    def _accept_test(self, lp_proposal: float, proposal: ArrayType) -> bool:
        return metropolis_accept_test(lp_proposal, self._log_p_theta, self._rng)
