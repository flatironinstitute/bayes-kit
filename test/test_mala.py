from test.models.binomial import Binomial
from test.models.std_normal import StdNormal
from bayes_kit.algorithms import MALA
import numpy as np


def test_mala_std_normal() -> None:
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    mala = MALA(model, 0.3, init)

    M = 10000
    draws = np.array([mala.step()[0] for _ in range(M)])

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)


def test_mala_binom() -> None:
    model = Binomial(alpha=2, beta=3, x=5, N=15)
    M = 1200
    mala = MALA(model, 0.07, init=np.array([model.initial_state(0)]))

    draws = model.constrain_draws(np.array([mala.step()[0] for _ in range(M)]))

    # skip 200 draws as a "burn-in" to try to make estimates less noisy
    mean = draws[200:].mean(axis=0)
    var = draws[200:].var(axis=0, ddof=1)

    print(f"{draws[1:10]=}")
    print(f"{mean=}  {var=}")
    print(f"acceptance rate : {1 - (draws[1:] == draws[:-1] ).sum() / M}")

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.05)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.01)


def test_mala_repr() -> None:
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()

    mala_1 = MALA(model, 0.3, init, seed=123)
    mala_2 = MALA(model, 0.3, init, seed=123)
    mala_3 = MALA(model, 0.3, init, seed=321)

    M = 25
    draws_1 = np.array([mala_1.step()[0] for _ in range(M)])
    draws_2 = np.array([mala_2.step()[0] for _ in range(M)])
    draws_3 = np.array([mala_3.step()[0] for _ in range(M)])

    np.testing.assert_array_equal(draws_1, draws_2)
    # Confirm that different results occur with different seeds
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(draws_1, draws_3)
