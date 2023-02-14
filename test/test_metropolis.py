from test.models.std_normal import StdNormal
from test.models.beta_binomial import BetaBinom
from test.models.skew_normal import SkewNormal
from bayes_kit.metropolis import (
    Metropolis,
    MetropolisHastings,
    metropolis_accept_test,
    metropolis_hastings_accept_test,
)
from unittest.mock import Mock

import numpy as np
import scipy.stats as sst
from numpy.typing import NDArray


def test_metropolis_accept_test_accepts_more_likely_proposal() -> None:
    mock_random = Mock()
    mock_random.uniform = Mock(
        return_value=1
    )  # exceeds theoretical and equals practical maximum return from rand.uniform()
    # for any value of lp_proposal > lp_current, even the maximum random draw should result in accepting the proposal
    lp_proposal = np.log(np.random.uniform())
    lp_current = 5
    while lp_current >= lp_proposal:
        lp_current = np.log(np.random.uniform())
    assert metropolis_accept_test(lp_proposal, lp_current, mock_random)


def test_metropolis_accept_test_rejects_when_proposal_below_uniform_draw() -> None:
    mock_random = Mock()
    mock_random.uniform = Mock(return_value=0.5)
    # if the uniform draw is 0.5, then we should reject the proposal if the current is at least 2x as likely as the proposal
    pr_proposal = 0.4
    pr_current = 0.80000
    lp_proposal = np.log(pr_proposal)
    lp_current = np.log(pr_current)
    assert not metropolis_accept_test(lp_proposal, lp_current, mock_random)


def test_metropolis_accept_test_accepts_when_proposal_above_uniform_draw() -> None:
    mock_random = Mock()
    mock_random.uniform = Mock(return_value=0.5)
    # if the uniform draw is 0.5, then we should accept the proposal if the current is less than 2x as likely as the proposal
    pr_proposal = 0.4
    pr_current = 0.7999999
    lp_proposal = np.log(pr_proposal)
    lp_current = np.log(pr_current)
    assert metropolis_accept_test(lp_proposal, lp_current, mock_random)


def test_metropolis_hastings_accept_test_accepts_more_likely_proposal_given_transition_ratio():
    mock_random = Mock()
    mock_random.uniform = Mock(return_value=0.5)
    pr_proposal = 0.4
    pr_current = (
        0.81  # This should cause rejection if the transition likelihood is symmetric
    )
    lp_proposal = np.log(pr_proposal)
    lp_current = np.log(pr_current)
    lp_transition_balanced = np.log(0.5)
    assert not metropolis_hastings_accept_test(
        lp_proposal,
        lp_current,
        lp_forward_transition=lp_transition_balanced,
        lp_reverse_transition=lp_transition_balanced,
        rand=mock_random,
    )
    # Now use asymmetric transition probabilities so that the reverse transition is
    # slightly more likely, thus "explaining away" that the current state is more
    # likely than the proposed state.
    lp_forward_transition = np.log(0.4)
    lp_reverse_transition = np.log(0.6)
    assert metropolis_hastings_accept_test(
        lp_proposal,
        lp_current,
        lp_forward_transition=lp_forward_transition,
        lp_reverse_transition=lp_reverse_transition,
        rand=mock_random,
    )


def test_metropolis_hastings_accept_test_reduces_to_metropolis_given_equal_transition_likelihoods():
    M = 1000
    mock_uniform = Mock()
    for _ in range(M):
        lp_proposal = np.log(np.random.uniform())
        lp_current = np.log(np.random.uniform())
        lp_transition = np.log(np.random.uniform())
        mock_uniform.uniform = Mock(return_value=np.random.uniform())
        metropolis_result = metropolis_accept_test(
            lp_proposal, lp_current, mock_uniform
        )
        metropolis_hastings_result = metropolis_hastings_accept_test(
            lp_proposal, lp_current, lp_transition, lp_transition, mock_uniform
        )
        assert metropolis_result == metropolis_hastings_result


def test_metropolis_std_normal() -> None:
    M = 2500
    model = StdNormal()
    # We generally don't want to set random seeds for tests, but here doing so lets us
    # set a tighter tolerance and lower iteration count while still avoiding stochastic failure.
    gen = np.random.default_rng(seed=12345)
    # init with draw from posterior
    init = gen.normal(loc=0, scale=1, size=[1])
    proposal_fn = lambda theta: gen.normal(loc=theta, scale=4)
    metropolis = Metropolis(model, proposal_fn=proposal_fn, init=init, seed=1777)

    draws = np.array([metropolis.sample()[0] for _ in range(M)])
    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)


def test_metropolis_beta_binom() -> None:
    M = 1000
    model = BetaBinom()
    init = np.array([model.initial_state(0)])
    proposal_fn = lambda theta: np.random.normal(loc=theta, scale=4)
    metropolis = Metropolis(model, proposal_fn=proposal_fn, init=init)

    draws = np.array([metropolis.sample()[0] for _ in range(M)])
    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)


def test_metropolis_hastings_skew_normal() -> None:
    M = 2500
    skewness_a = 4
    model = SkewNormal(a=skewness_a)
    # We don't generally want to set random seeds for tests, but this sampler converges quite slowly.
    # So the options are either fixing the seed, setting an unbearably high iteration count, setting
    # a very soft tolerance, or accepting stochastic test failure. Among those choices,
    # a test that has tight tolerance on a known value in a tolerable runtime seems most likely to
    # actually inform us if any bugs are introduced in the code.
    gen = np.random.default_rng(seed=1701)
    init: NDArray[np.float64] = sst.skewnorm.rvs(skewness_a, size=[1], random_state=gen)  # type: ignore
    p_fn = lambda theta: np.array(
        [sst.skewnorm.rvs(skewness_a, loc=theta, random_state=gen)]
    )
    trns_lp_fn = lambda observation, given: sst.skewnorm.logpdf(
        observation, skewness_a, loc=given
    )[0]
    mh = MetropolisHastings(
        model, proposal_fn=p_fn, transition_lp_fn=trns_lp_fn, init=init, seed=1777
    )

    draws = np.array([mh.sample()[0] for _ in range(M)])
    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.18)


def test_metropolis_reproducible() -> None:
    model = StdNormal()
    init = np.random.normal(loc=0, scale=1, size=[1])
    M = 25

    proposal_generator = np.random.default_rng(seed=12345)
    proposal_fn = lambda theta: proposal_generator.normal(loc=theta, scale=4)
    metropolis_1 = Metropolis(model, proposal_fn=proposal_fn, init=init, seed=1848)
    # Each model needs to be sampled from before instantiating the next, or they
    # trip over each other's pRNG trajectories
    draws_1 = np.array([metropolis_1.sample()[0] for _ in range(M)])

    # reinitialize proposal generator for each model
    proposal_generator = np.random.default_rng(seed=12345)
    proposal_fn = lambda theta: proposal_generator.normal(loc=theta, scale=4)
    metropolis_2 = Metropolis(model, proposal_fn=proposal_fn, init=init, seed=1848)
    draws_2 = np.array([metropolis_2.sample()[0] for _ in range(M)])

    proposal_generator = np.random.default_rng(seed=12345)
    proposal_fn = lambda theta: proposal_generator.normal(loc=theta, scale=4)
    metropolis_3 = Metropolis(model, proposal_fn=proposal_fn, init=init, seed=1912)
    draws_3 = np.array([metropolis_3.sample()[0] for _ in range(M)])

    np.testing.assert_array_equal(draws_1, draws_2)
    # Now confirm that results do not match when a different seed is used
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(draws_1, draws_3)


# Used in testing metropolis-hastings reproducibility.
# This winds up being more stable than rewriting the lambda directly in the test fn body.
def skewnorm_proposal_fn_factory(skewness: float, gen: np.random.Generator):
    fn = lambda t: np.array([sst.skewnorm.rvs(skewness, loc=t, random_state=gen)])
    return fn


def test_metropolis_hastings_reproducible() -> None:
    skewness_a = 4
    model = SkewNormal(a=skewness_a)
    M = 50
    g = np.random.default_rng(seed=123)
    init: NDArray[np.float64] = sst.skewnorm.rvs(skewness_a, size=[1], random_state=g)  # type: ignore
    transition_lp_fn = lambda o, g: sst.skewnorm.logpdf(o, skewness_a, loc=g)[0]

    proposal_generator = np.random.default_rng(seed=12345)
    proposal_fn = skewnorm_proposal_fn_factory(skewness_a, proposal_generator)
    mh_1 = MetropolisHastings(
        model,
        proposal_fn=proposal_fn,
        transition_lp_fn=transition_lp_fn,
        init=init,
        seed=1848,
    )
    draws_1 = np.array([mh_1.sample()[0] for _ in range(M)])

    proposal_generator = np.random.default_rng(seed=12345)
    proposal_fn = skewnorm_proposal_fn_factory(skewness_a, proposal_generator)
    mh_2 = MetropolisHastings(
        model,
        proposal_fn=proposal_fn,
        transition_lp_fn=transition_lp_fn,
        init=init,
        seed=1848,
    )
    draws_2 = np.array([mh_2.sample()[0] for _ in range(M)])

    proposal_generator = np.random.default_rng(seed=12345)
    proposal_fn = skewnorm_proposal_fn_factory(skewness_a, proposal_generator)
    mh_3 = MetropolisHastings(
        model,
        proposal_fn=proposal_fn,
        transition_lp_fn=transition_lp_fn,
        init=init,
        seed=518,
    )
    draws_3 = np.array([mh_3.sample()[0] for _ in range(M)])

    np.testing.assert_array_equal(draws_1, draws_2)
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(draws_1, draws_3)

def test_metropolis_hastings_iter_returns_self() -> None:
    model = StdNormal()
    proposal_fn = lambda x: 1
    transition_lp_fn = lambda x, y: 1
    mh = MetropolisHastings(model, proposal_fn, transition_lp_fn)
    i = iter(mh)
    assert i == mh

def test_metropolis_hastings_next_trajectory_matches_calling_sample() -> None:
    model = StdNormal()
    init = np.random.normal(loc=0, scale=1, size=[1])
    M = 25

    proposal_generator = np.random.default_rng(seed=123)
    p_fn = lambda theta: proposal_generator.normal(loc=theta, scale=4)
    x_fn = lambda o, g: 1  # symmetric likelihood--don't bother to compute
    mh1 = MetropolisHastings(
        model, proposal_fn=p_fn, transition_lp_fn=x_fn, init=init, seed=996
    )
    draws_1 = np.array([mh1.sample()[0] for _ in range(M)])

    proposal_generator = np.random.default_rng(seed=123)
    p_fn = lambda theta: proposal_generator.normal(loc=theta, scale=4)
    mh2 = MetropolisHastings(
        model, proposal_fn=p_fn, transition_lp_fn=x_fn, init=init, seed=996
    )
    draws_2 = np.array([next(mh2)[0] for _ in range(M)])

    np.testing.assert_array_equal(draws_1, draws_2)


def test_metropolis_hastings_throws_when_proposal_fn_generates_wrong_type() -> None:
    model = StdNormal()
    proposal_fn = lambda x: "a"
    transition_lp_fn = lambda x, y: 1
    mh = MetropolisHastings(model, proposal_fn, transition_lp_fn)
    with np.testing.assert_raises(ValueError):
        _ = mh.sample()
