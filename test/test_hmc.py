import functools
from test.models.binomial import Binomial
from test.models.std_normal import StdNormal
from typing import Any

import numpy as np
import pytest

from bayes_kit.hmc import HMCDiag


def _call_counter(f: Any) -> Any:
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.calls += 1  # type: ignore
        return f(*args, **kwargs)

    wrapper.calls = 0  # type: ignore
    return wrapper


@pytest.mark.parametrize("steps", [0, 1, 10])
def test_hmc_leapfrog_num_evals(steps: int) -> None:
    # Expect that HMC.leapfrog calls log_density only once per step
    model = StdNormal()
    model.log_density = _call_counter(model.log_density)  # type: ignore
    model.log_density_gradient = _call_counter(model.log_density_gradient)  # type: ignore

    hmc = HMCDiag(model, steps=steps, stepsize=0.25)
    _ = hmc.sample()

    # Expect one call to log_density before leapfrog and one after
    assert model.log_density.calls == 2  # type: ignore
    # Expect one call to log_density_gradient per leapfrog step, plus one for initial/final half step
    assert model.log_density_gradient.calls == hmc._steps + 1  # type: ignore


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


def test_hmc_binom() -> None:
    model = Binomial(alpha=2, beta=3, x=5, N=15)
    M = 800
    hmc = HMCDiag(
        model, stepsize=0.08, steps=3, init=np.array([model.initial_state(0)])
    )

    draws = model.constrain_draws(np.array([hmc.sample()[0] for _ in range(M)]))

    # skip 100 draws as a "burn-in" to try to make estimates less noisy
    mean = draws[100:].mean(axis=0)
    var = draws[100:].var(axis=0, ddof=1)

    print(f"{draws[1:10]=}")
    print(f"{mean=}  {var=}")
    print(f"acceptance rate : {1 - (draws[1:] == draws[:-1] ).sum() / M}")

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.05)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.01)
