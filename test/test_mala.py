from test.models.std_normal import StdNormal
from test.models.beta_binomial import BetaBinom
from bayes_kit.mala import MALA
import numpy as np


def test_mala_std_normal() -> None:
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    mala = MALA(model, 0.3, init)

    M = 10000
    draws = np.array([mala.sample()[0] for _ in range(M)])

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)


def test_mala_beta_binom() -> None:
    model = BetaBinom()
    M = 1000
    mala = MALA(model, 0.5, init=np.array([model.initial_state(0)]))

    draws = np.array([mala.sample()[0] for _ in range(M)])

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    print(f"{draws[1:10]=}")
    print(f"{mean=}  {var=}")

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.05)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.005)
