from test.models.std_normal import StdNormal
from bayes_kit.hmc import HMCDiag
import numpy as np


def test_hmc_diag_std_normal() -> None:
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    hmc = HMCDiag(model, steps=10, stepsize=0.25, init=init)

    M = 10000
    draws = np.array([hmc.sample()[0] for _ in range(M)])

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)

def test_hmc_diag_repr() -> None:
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()

    hmc_1 = HMCDiag(model, steps=10, stepsize=0.25, init=init, seed=123)
    hmc_2 = HMCDiag(model, steps=10, stepsize=0.25, init=init, seed=123)

    M = 25
    draws_1 = np.array([hmc_1.sample()[0] for _ in range(M)])
    draws_2 = np.array([hmc_2.sample()[0] for _ in range(M)])

    np.testing.assert_array_equal(draws_1, draws_2)
