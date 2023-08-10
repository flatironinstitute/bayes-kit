from test.models.binomial import Binomial
from test.models.std_normal import StdNormal
from bayes_kit.drghmc import DrGhmcDiag
import numpy as np
import functools
import re
import pytest


def _call_counter(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


def upper_bound_leapfrog_steps(stepcounts):
    """Upper bound on the number of leapfrog steps used to generate a single sample.

    Computing the exact number of leapfrog steps is difficult because (1) we don't
    know how many proposal attempts the sampler makes and (2) the accept() function
    sometimes terminates early (this occurs when the probability of accepting a ghost
    point is 1).

    Instead, we compute an upper bound on the number of leapfrog steps: assume that
    the sampler makes the maximum number of proposals and the accept() function never
    terminates early.

    Args:
        stepcounts: list of number of leapfrog steps

    Returns:
        upper bound on number of leapfrog steps
    """
    ub_steps = 0
    for idx, stepcount in enumerate(reversed(stepcounts)):
        # Leapfrog integration with `stepcount` number of steps is performed multiple
        # times when computing ghost samples.
        repetitions = 2 ** (idx)
        ub_steps += stepcount * repetitions
    return ub_steps


@pytest.mark.parametrize("proposals", [1, 2, 3, 4, 5])
def test_drghmc_num_grad_evals_one_sample(proposals) -> None:
    model = StdNormal()
    model.log_density_gradient = _call_counter(model.log_density_gradient)  # type: ignore

    drghmc = DrGhmcDiag(
        model=model,
        proposals=proposals,
        leapfrog_stepsizes=[1.0] * proposals,
        leapfrog_stepcounts=[2 * (i + 1) for i in range(proposals)],
        damping=0.001,
        prob_retry=True,
    )
    _ = drghmc.sample()

    ub_steps = upper_bound_leapfrog_steps(drghmc._leapfrog_stepcounts)

    # Expect one call to log_density_gradient per leapfrog step, plus one for
    # calculating the density and gradient of the intial sample.
    assert model.log_density_gradient.calls <= 1 + ub_steps  # type: ignore


@pytest.mark.parametrize("num_samples", [1, 2, 3, 10])
def test_drghmc_num_grad_evals_many_samples(num_samples) -> None:
    model = StdNormal()
    model.log_density_gradient = _call_counter(model.log_density_gradient)  # type: ignore

    drghmc = DrGhmcDiag(
        model=model,
        proposals=1,
        leapfrog_stepsizes=[95.0],
        leapfrog_stepcounts=[1],
        damping=0.001,
        prob_retry=True,
    )
    _ = np.array([drghmc.sample()[0] for _ in range(num_samples)])

    ub_steps_one_sample = upper_bound_leapfrog_steps(drghmc._leapfrog_stepcounts)
    ub_total_steps = ub_steps_one_sample * num_samples

    # Expect one call to log_density_gradient per leapfrog step, plus one for
    # calculating the density and gradient of the intial sample.
    assert model.log_density_gradient.calls <= 1 + ub_total_steps  # type: ignore


def test_drghmc_diag_std_normal() -> None:
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    drghmc = DrGhmcDiag(
        model,
        proposals=3,
        leapfrog_stepcounts=[10, 10 * 2, 10 * 4],
        leapfrog_stepsizes=[0.25, 0.25 / 2, 0.25 / 4],
        damping=0.9,
        init=init,
    )

    M = 10000
    draws = np.array([drghmc.sample()[0] for _ in range(M)])

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)


def test_drghmc_diag_repr() -> None:
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()

    drghmc_1 = DrGhmcDiag(
        model,
        proposals=3,
        leapfrog_stepcounts=[10, 20, 30],
        leapfrog_stepsizes=[0.25, 0.25 / 4, 0.25 / 8],
        damping=0.2,
        init=init,
        seed=123,
    )

    drghmc_2 = DrGhmcDiag(
        model,
        proposals=3,
        leapfrog_stepcounts=[10, 20, 30],
        leapfrog_stepsizes=[0.25, 0.25 / 4, 0.25 / 8],
        damping=0.2,
        init=init,
        seed=123,
    )

    M = 25
    draws_1 = np.array([drghmc_1.sample()[0] for _ in range(M)])
    draws_2 = np.array([drghmc_2.sample()[0] for _ in range(M)])

    np.testing.assert_array_equal(draws_1, draws_2)


def test_drghmc_binom() -> None:
    model = Binomial(alpha=2, beta=3, x=5, N=15)
    proposals = 3
    M = 800

    drghmc = DrGhmcDiag(
        model,
        proposals,
        leapfrog_stepsizes=[0.1] * proposals,
        leapfrog_stepcounts=[3] * proposals,
        damping=0.2,
        init=np.array([model.initial_state(0)]),
    )

    draws = model.constrain_draws(np.array([drghmc.sample()[0] for _ in range(M)]))

    # skip 100 draws as a "burn-in" to try to make estimates less noisy
    mean = draws[100:].mean(axis=0)
    var = draws[100:].var(axis=0, ddof=1)

    print(f"{draws[1:10]=}")
    print(f"{mean=}  {var=}")
    print(f"acceptance rate : {1 - (draws[1:] == draws[:-1] ).sum() / M}")

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.05)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.01)


def test_drghmc_invalid_proposals() -> None:
    model = StdNormal()

    def drghmc_proposals(proposals):
        drghmc = DrGhmcDiag(
            model=model,
            proposals=proposals,
            leapfrog_stepsizes=[1],
            leapfrog_stepcounts=[1],
            damping=0.001,
            prob_retry=True,
        )
        return drghmc

    invalid_proposals = [1]
    err_message = f"proposals must be an int, not {type(invalid_proposals)}"
    with pytest.raises(TypeError, match=err_message):
        drghmc_proposals(invalid_proposals)

    invalid_proposals = 1.0
    err_message = f"proposals must be an int, not {type(invalid_proposals)}"
    with pytest.raises(TypeError, match=err_message):
        drghmc_proposals(invalid_proposals)

    invalid_proposals = 0
    err_message = (
        f"proposals must be greater than or equal to 1, not {invalid_proposals}"
    )
    with pytest.raises(ValueError, match=err_message):
        drghmc_proposals(invalid_proposals)

    invalid_proposals = -1
    err_message = (
        f"proposals must be greater than or equal to 1, not {invalid_proposals}"
    )
    with pytest.raises(ValueError, match=err_message):
        drghmc_proposals(invalid_proposals)


def test_drghmc_invalid_damping() -> None:
    model = StdNormal()

    def drghmc_proposals(damping):
        proposals = 3
        drghmc = DrGhmcDiag(
            model=model,
            proposals=proposals,
            leapfrog_stepsizes=[1.0] * proposals,
            leapfrog_stepcounts=[1] * proposals,
            damping=damping,
            prob_retry=True,
        )
        return drghmc

    invalid_damping = [0.5]
    err_message = f"damping must be a float, not {type(invalid_damping)}"
    with pytest.raises(TypeError, match=err_message):
        drghmc_proposals(invalid_damping)

    invalid_damping = int(1)
    err_message = f"damping must be a float, not {type(invalid_damping)}"
    with pytest.raises(TypeError, match=err_message):
        drghmc_proposals(invalid_damping)

    invalid_damping = float(0)
    err_message = f"damping of {invalid_damping} must be within (0, 1]"
    with pytest.raises(ValueError, match=re.escape(err_message)):
        drghmc_proposals(invalid_damping)

    invalid_damping = float(-1)
    err_message = f"damping of {invalid_damping} must be within (0, 1]"
    with pytest.raises(ValueError, match=re.escape(err_message)):
        drghmc_proposals(invalid_damping)


def test_drghmc_invalid_leapfrog_stepsizes() -> None:
    model = StdNormal()

    def drghmc_stepsizes(stepsizes, proposals):
        drghmc = DrGhmcDiag(
            model=model,
            proposals=proposals,
            leapfrog_stepsizes=stepsizes,
            leapfrog_stepcounts=[1] * proposals,
            damping=0.001,
            prob_retry=True,
        )
        return drghmc

    proposals = 1
    invalid_stepsizes = 1
    err_message = f"leapfrog_stepsizes must be a list, not {type(invalid_stepsizes)}"
    with pytest.raises(TypeError, match=err_message):
        drghmc_stepsizes(invalid_stepsizes, proposals)

    proposals = 1
    invalid_stepsizes = 0.25
    err_message = f"leapfrog_stepsizes must be a list, not {type(invalid_stepsizes)}"
    with pytest.raises(TypeError, match=err_message):
        drghmc_stepsizes(invalid_stepsizes, proposals)

    proposals = 2
    invalid_stepsizes = [0.25, 0.25, 0.25]
    err_message = (
        f"leapfrog_stepsizes must be a list of length {proposals}, not length "
        f"{len(invalid_stepsizes)}, so that each proposal has a specified leapfrog "
        f"stepsize"
    )
    with pytest.raises(ValueError, match=err_message):
        drghmc_stepsizes(invalid_stepsizes, proposals)

    proposals = 4
    invalid_stepsizes = [0.25, 0.25, 0.25]
    err_message = (
        f"leapfrog_stepsizes must be a list of length {proposals}, not length "
        f"{len(invalid_stepsizes)}, so that each proposal has a specified leapfrog "
        f"stepsize"
    )
    with pytest.raises(ValueError, match=err_message):
        drghmc_stepsizes(invalid_stepsizes, proposals)

    proposals = 2
    invalid_stepsizes = [float(1), int(1)]
    invalid_idx = 1
    err_message = (
        f"leapfrog stepsizes must be of type float, not "
        f"{type(invalid_stepsizes[invalid_idx])} at index {invalid_idx}"
    )
    with pytest.raises(TypeError, match=err_message):
        drghmc_stepsizes(invalid_stepsizes, proposals)

    proposals = 3
    invalid_stepsizes = [0.25, 0.24, 0.25]
    invalid_idx = 1
    invalid_cur = invalid_stepsizes[invalid_idx + 1]
    invalid_prev = invalid_stepsizes[invalid_idx]
    err_message = (
        f"leapfrog stepsizes must be non-increasing, but found stepsize of "
        f"{invalid_cur} at index {invalid_idx + 1} which is greater than stepsize of "
        f"{invalid_prev} at index {invalid_idx}"
    )

def test_drghmc_invalid_leapfrog_stepcounts() -> None:
    model = StdNormal()

    def drghmc_stepcounts(stepcounts, proposals):
        drghmc = DrGhmcDiag(
            model=model,
            proposals=proposals,
            leapfrog_stepsizes=[1.] * proposals,
            leapfrog_stepcounts=stepcounts,
            damping=0.001,
            prob_retry=True,
        )
        return drghmc

    proposals = 1
    invalid_stepcounts = 1
    err_message = f"leapfrog_stepcounts must be a list, not {type(invalid_stepcounts)}"
    with pytest.raises(TypeError, match=err_message):
        drghmc_stepcounts(invalid_stepcounts, proposals)

    proposals = 1
    invalid_stepcounts = 0.25
    err_message = f"leapfrog_stepcounts must be a list, not {type(invalid_stepcounts)}"
    with pytest.raises(TypeError, match=err_message):
        drghmc_stepcounts(invalid_stepcounts, proposals)

    proposals = 2
    invalid_stepcounts = [0.25, 0.25, 0.25]
    err_message = (
        f"leapfrog_stepcounts must be a list of length {proposals}, not length "
        f"{len(invalid_stepcounts)}, so that each proposal has a specified number of "
        f"leapfrog steps"
    )
    with pytest.raises(ValueError, match=err_message):
        drghmc_stepcounts(invalid_stepcounts, proposals)

    proposals = 4
    invalid_stepcounts = [0.25, 0.25, 0.25]
    err_message = (
        f"leapfrog_stepcounts must be a list of length {proposals}, not length "
        f"{len(invalid_stepcounts)}, so that each proposal has a specified number of "
        f"leapfrog steps"
    )
    with pytest.raises(ValueError, match=err_message):
        drghmc_stepcounts(invalid_stepcounts, proposals)

    proposals = 2
    invalid_stepcounts = [1, 3.5]
    invalid_idx = 1
    err_message = (
        f"leapfrog stepcounts must be of type int, not "
        f"{type(invalid_stepcounts[invalid_idx])} at index {invalid_idx}"
    )
    with pytest.raises(TypeError, match=err_message):
        drghmc_stepcounts(invalid_stepcounts, proposals)
