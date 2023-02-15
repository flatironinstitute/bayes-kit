"""Test different algorithms with known equivalencies against each other"""

from test.models.std_normal import StdNormal
from test.models.skew_normal import SkewNormal
from bayes_kit import MALA, HMCDiag, Metropolis, MetropolisHastings
import numpy as np
import scipy.stats as sst


def test_hmc_mala_agreement() -> None:
    """HMC with 1 step is equivalent to MALA"""
    model = StdNormal()
    init = np.array([0.2])

    epsilon_hmc = 0.02
    hmc = HMCDiag(model, stepsize=epsilon_hmc, steps=1, init=init, seed=123)

    # HMC and MALA have different interpretations of stepsize parameter epsilon
    # HMC:  theta_proposal = theta  +  epsilon * std_normal()            +  1/2 * epsilon ^ 2 * grad(theta)
    # MALA: theta_proposal = theta  +  sqrt(2 * epsilon) * std_normal()  +  epsilon * grad(theta)
    # So we can make them equivalent by setting epsilon_mala = 1/2 * epsilon_hmc ^ 2
    mala = MALA(model, epsilon=0.5 * epsilon_hmc**2, init=init, seed=123)

    M = 50
    draws_1 = np.array([hmc.sample()[0] for _ in range(M)])
    draws_2 = np.array([mala.sample()[0] for _ in range(M)])

    np.testing.assert_array_almost_equal(draws_1, draws_2)
    # make sure we didn't just stay at the initial value forever
    assert len(np.unique(draws_1)) > 20

def test_metropolis_hastings_reduces_to_metropolis() -> None:
    """Metropolis Hastings is equivalent to Metropolis when the proposal is symmetric"""

    M = 25
    skewness_a = 0
    model = SkewNormal(a=skewness_a)  # which is 0, i.e. standard normal
    init = np.random.normal(loc=0, scale=1, size=[1])
    t_lp_fn = lambda o, g: sst.skewnorm.logpdf(o, skewness_a, loc=g)[0]

    gen = np.random.default_rng(seed=12345)
    proposal_fn = lambda t: np.array(
        [sst.skewnorm.rvs(skewness_a, loc=t, random_state=gen)]
    )
    metropolis = Metropolis(model, proposal_fn=proposal_fn, init=init, seed=1848)
    draws_from_metropolis = np.array([metropolis.sample()[0] for _ in range(M)])

    gen = np.random.default_rng(seed=12345)
    proposal_fn = lambda t: np.array(
        [sst.skewnorm.rvs(skewness_a, loc=t, random_state=gen)]
    )
    mh = MetropolisHastings(
        model, proposal_fn=proposal_fn, transition_lp_fn=t_lp_fn, init=init, seed=1848
    )
    draws_from_mh = np.array([mh.sample()[0] for _ in range(M)])

    np.testing.assert_array_equal(draws_from_metropolis, draws_from_mh)
