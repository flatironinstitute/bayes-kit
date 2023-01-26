from test.models.std_normal import StdNormal
from bayes_kit.rwm import RandomWalkMetropolis
import numpy as np


def test_rwm_std_normal() -> None:
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    proposal_rng = lambda theta: np.random.normal(loc=theta, scale=4)
    rwm = RandomWalkMetropolis(model, proposal_rng, init)
    M = 10000
    draws = np.array([rwm.sample()[0] for _ in range(M)])
    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)
    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)

    accept = M - (draws[: M - 1] == draws[1:]).sum()
    print(f"{accept=}")
    print(f"{draws[1:10]=}")
    print(f"{mean=}  {var=}")


def test_rwm_repr() -> None:

    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()

    prop_gen_1 = np.random.default_rng(123)
    proposal_1 = lambda theta: prop_gen_1.normal(loc=theta, scale=4)
    rwm_1 = RandomWalkMetropolis(model, proposal_1, init, seed=456)

    prop_gen_2 = np.random.default_rng(123)
    proposal_2 = lambda theta: prop_gen_2.normal(loc=theta, scale=4)
    rwm_2 = RandomWalkMetropolis(model, proposal_2, init, seed=456)

    M = 25
    draws_1 = np.array([rwm_1.sample()[0] for _ in range(M)])
    draws_2 = np.array([rwm_2.sample()[0] for _ in range(M)])

    np.testing.assert_array_equal(draws_1, draws_2)
