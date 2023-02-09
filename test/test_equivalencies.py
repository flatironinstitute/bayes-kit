"""Test different algorithms with known equivalencies against each other"""

from test.models.beta_binomial import BetaBinom
from bayes_kit import MALA, HMCDiag
import numpy as np


def test_hmc_mala_agreement():
    """HMC with 1 step is equivalent to MALA"""
    model = BetaBinom()
    init = np.array([model.initial_state(0)])

    epsilon = 0.02
    hmc = HMCDiag(model, epsilon, steps=1, init=init, seed=123)

    # Different interpretations of stepsize parameter epsilon
    # in HMC,  theta_proposal = theta   + epsilon * std_normal()              + 1/2 * epsilon ^ 2 * grad(theta)
    # in MALA, theta_proposal = theta   + sqrt(2 * epsilon) * std_normal()    + epsilon * grad(theta)
    # So we can make them equivalent by setting epsilon_mala = 1/2 * epsilon_hmc ^ 2
    epsilon_mala = 0.5 * epsilon**2
    mala = MALA(model, epsilon_mala, init=init, seed=123)

    M = 50
    draws_1 = np.array([hmc.sample()[0] for _ in range(M)])
    draws_2 = np.array([mala.sample()[0] for _ in range(M)])

    print(draws_1)
    print(draws_2)

    np.testing.assert_array_almost_equal(draws_1, draws_2)
