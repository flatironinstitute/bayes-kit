from test.models.binomial import Binomial
from test.models.std_normal import StdNormal
from bayes_kit.drhmc import PdrGhmcDiag
import numpy as np
import functools
import pytest


def _call_counter(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


@pytest.mark.parametrize("num_proposals", [1, 2, 3])
def test_drhmc_leapfrog_num_evals(num_proposals) -> None:
    model = StdNormal()
    model.log_density = _call_counter(model.log_density)
    model.log_density_gradient = _call_counter(model.log_density_gradient)

    drhmc = PdrGhmcDiag(model, steps=10, stepsize=0.25, num_proposals=num_proposals)
    drhmc.accept = _call_counter(drhmc.accept)
    _ = drhmc.sample()

    # Evaluate less than 2 ** num_proposals acceptance probablities as per DRHMC paper
    assert drhmc.accept.calls < np.power(2, drhmc._num_proposals)
    # Expect one call to log_density per leapfrog function, called by accept() function
    assert model.log_density.calls == 1 + drhmc.accept.calls
    # Expect one call to log_density_gradient per leapfrog step, plus one for initial/final half step
    assert (
        model.log_density_gradient.calls
        == (drhmc._steps_list[0] + 1) * drhmc.accept.calls
    )


@pytest.mark.parametrize("adaptivity_factor", [1, 2, 5])
def test_drhmc_std_normal_adaptive(adaptivity_factor) -> None:
    model = StdNormal()
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])

    stepsize = lambda k: 0.25 * (adaptivity_factor**-k)
    steps = lambda k: 10 * (adaptivity_factor**k)
    drhmc = PdrGhmcDiag(model, steps=steps, stepsize=stepsize, init=init)

    M = 10000
    draws = np.array([drhmc.sample()[0] for _ in range(M)])

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)


def test_drhmc_diag_repr() -> None:
    model = StdNormal()
    rng = np.random.default_rng(seed=123)
    init = rng.normal(size=model.dims())

    drhmc_1 = PdrGhmcDiag(model, steps=10, stepsize=0.25, init=init, seed=123)
    drhmc_2 = PdrGhmcDiag(model, steps=10, stepsize=0.25, init=init, seed=123)

    M = 25
    draws_1 = np.array([drhmc_1.sample()[0] for _ in range(M)])
    draws_2 = np.array([drhmc_2.sample()[0] for _ in range(M)])

    np.testing.assert_array_equal(draws_1, draws_2)


@pytest.mark.parametrize("adaptivity_factor", [1, 2, 5])
def test_drhmc_binom_adaptive(adaptivity_factor) -> None:
    model = Binomial(alpha=2, beta=3, x=5, N=15)
    init = np.array([model.initial_state(0)])

    stepsize = lambda k: 0.08 * (adaptivity_factor**-k)
    steps = lambda k: 3 * (adaptivity_factor**k)
    drhmc = PdrGhmcDiag(model, stepsize=stepsize, steps=steps, init=init)

    M = 800
    draws = model.constrain_draws(np.array([drhmc.sample()[0] for _ in range(M)]))

    # skip 100 draws as a "burn-in" to try to make estimates less noisy
    mean = draws[100:].mean(axis=0)
    var = draws[100:].var(axis=0, ddof=1)

    print(f"{draws[:10]=}")
    print(f"{mean=}  {var=}")
    print(f"acceptance rate : {1 - (draws[1:] == draws[:-1] ).sum() / M}")

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.05)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.01)


@pytest.mark.parametrize("init_stepsize", [-2, -1, 0])
def test_drhmc_invalid_stepsize(init_stepsize) -> None:
    model = StdNormal()
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])

    adaptivity_factor = 2
    stepsize = lambda k: init_stepsize * (adaptivity_factor**-k)
    steps = lambda k: 10 * (adaptivity_factor**k)

    with pytest.raises(ValueError) as excinfo:
        drhmc = PdrGhmcDiag(model, steps=steps, stepsize=stepsize, init=init)
        M = 10000
        draws = np.array([drhmc.sample()[0] for _ in range(M)])

    assert (
        str(excinfo.value)
        == f"stepsize must be positive, but found stepsize of {init_stepsize}"
    )


@pytest.mark.parametrize("init_steps", [-2, -1, 0, 0.5])
def test_drhmc_invalid_steps(init_steps) -> None:
    model = StdNormal()
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])

    adaptivity_factor = 2
    stepsize = lambda k: 0.25 * (adaptivity_factor**-k)
    steps = lambda k: init_steps * (adaptivity_factor**k)

    with pytest.raises((TypeError, ValueError)) as excinfo:
        drhmc = PdrGhmcDiag(model, steps=steps, stepsize=stepsize, init=init)
        M = 10000
        draws = np.array([drhmc.sample()[0] for _ in range(M)])

    if excinfo.type == ValueError:
        assert (
            str(excinfo.value)
            == f"steps must be positive, but found {init_steps} steps"
        )
    elif excinfo.type == TypeError:
        assert (
            str(excinfo.value)
            == f"callable must return integers number of steps, but found {type(init_steps)} steps"
        )
