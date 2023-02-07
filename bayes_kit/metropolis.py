from typing import Callable, Iterator, Optional, Union
from numpy.typing import NDArray, ArrayLike
import numpy as np

from .model_types import LogDensityModel

# TODO: Add to global type definitions
Sample = tuple[NDArray[np.float64], float]
TransitionLPFn = Callable[
    [NDArray[np.float64], NDArray[np.float64]], float
]  # s.b. (TO_STATE, FROM_STATE) -> log_prob (float)
# TODO: make some decision about typing for parameters


def metropolis_accept_test(
    lp_proposal: float, lp_current: float, rand: np.random.Generator
) -> bool:
    """
    Return whether to accept a proposed new state.

    The Metropolis acceptance condition compares the likelihood of the proposal with the likelihood
    of the current value, as a ratio L(proposal)/L(current). The proposal is accepted if this ratio
    exceeds a uniformly-random draw from [0, 1) (so, acceptance is proportional to the likelihood ratio).

    Parameters:
        lp_proposal (float): log probability of the proposed parameter values (theta-star)
        lp_current (float): log probability of the current parameter values (theta)
        rand (np.random.Generator): an appropriately seeded pseudorandom number generator

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
    log_uniform_random: float = np.log(rand.uniform())
    return log_uniform_random < log_acceptance_ratio


def metropolis_hastings_accept_test(
    lp_proposal: float,
    lp_current: float,
    lp_forward_transition: float,
    lp_reverse_transition: float,
    rand: np.random.Generator,
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
        rand (np.random.Generator): an appropriately seeded pseudorandom number generator

    Returns:
        True if the proposal is accepted; false otherwise
    """
    # Since we have log likelihoods, replace multiplication/division with addition/subtraction.
    log_acceptance_ratio = lp_proposal - lp_current
    log_transition_likelihood_ratio = lp_reverse_transition - lp_forward_transition
    adjusted_log_acceptance_ratio = (
        log_acceptance_ratio + log_transition_likelihood_ratio
    )
    log_uniform_random: float = np.log(rand.uniform())
    return log_uniform_random < adjusted_log_acceptance_ratio


class MetropolisHastings:
    def __init__(
        self,
        model: LogDensityModel,
        proposal_fn: Callable[[NDArray[np.float64]], ArrayLike],
        transition_lp_fn: TransitionLPFn,
        *,
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        self._model = model
        self._dim = self._model.dims()
        self._rand = np.random.default_rng(seed)
        self._proposal_fn = (
            proposal_fn  # TODO: type check this once during construction?
        )
        self._transition_lp_fn = transition_lp_fn
        self._theta = init or self._rand.normal(size=self._dim)
        self._log_p_theta = self._model.log_density(self._theta)

    def __iter__(self) -> Iterator[Sample]:
        return self

    def __next__(self) -> Sample:
        return self.sample()

    def sample(self) -> Sample:
        proposal, lp_proposal = self._propose()
        accepted = self._accept_test(lp_proposal, proposal)
        if accepted:
            self._update_theta(proposal, lp_proposal)
        return (self._theta, self._log_p_theta)

    def _propose(self) -> Sample:
        untyped_proposed_theta = np.asanyarray(self._proposal_fn(self._theta))
        if not (untyped_proposed_theta.dtype == np.float64):
            raise TypeError(
                f"Proposal function must return a double; returned {untyped_proposed_theta.dtype=}."
            )
        proposed_theta: NDArray[np.float64] = untyped_proposed_theta
        lp_proposed_theta = self._model.log_density(proposed_theta)
        return (proposed_theta, lp_proposed_theta)

    def _update_theta(self, theta: NDArray[np.float64], log_p_theta: float) -> None:
        self._theta = theta
        self._log_p_theta = log_p_theta

    def _accept_test(self, lp_proposal: float, proposal: NDArray[np.float64]) -> bool:
        lp_forward_transition = self._transition_lp_fn(proposal, self._theta)
        lp_reverse_transition = self._transition_lp_fn(self._theta, proposal)
        return metropolis_hastings_accept_test(
            lp_proposal,
            self._log_p_theta,
            lp_forward_transition,
            lp_reverse_transition,
            self._rand,
        )


class Metropolis(MetropolisHastings):
    def __init__(
        self,
        model: LogDensityModel,
        proposal_fn: Callable[[NDArray[np.float64]], ArrayLike],
        *,
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        # This transition function will never be used--it isn't needed for Metropolis,
        # for which the transition probabilities are symmetric. But we need a valid one
        # for the superclass' constructor.
        dummy_transition_fn: TransitionLPFn = lambda a, b: 1
        super().__init__(model, proposal_fn, dummy_transition_fn, init=init, seed=seed)

    # 'proposal' isn't used, but we need signature consistency to override the parent method
    def _accept_test(self, lp_proposal: float, proposal: NDArray[np.float64]) -> bool:
        return metropolis_accept_test(lp_proposal, self._log_p_theta, self._rand)
