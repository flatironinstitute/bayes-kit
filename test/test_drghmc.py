import functools
import itertools
import re
from collections.abc import Sequence
from test.models.binomial import Binomial
from test.models.std_normal import StdNormal
from typing import Any

import numpy as np
import pytest

from bayes_kit.drghmc import DrGhmcDiag


def _call_counter(f: Any) -> Any:
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.calls += 1  # type: ignore
        return f(*args, **kwargs)

    wrapper.calls = 0  # type: ignore
    return wrapper


def upper_bound_leapfrog_steps(step_counts: Sequence[int]) -> int:
    """Upper bound on the number of leapfrog steps used to generate a single sample.

    Computing the exact number of leapfrog steps is difficult because (1) we don't
    know how many proposal attempts the sampler makes and (2) the accept() function
    sometimes terminates early (this occurs when the probability of accepting a ghost
    proposal is 1).

    Instead, we compute an upper bound on the number of leapfrog steps: assume that
    the sampler makes the maximum number of proposals and the accept() function never
    terminates early.

    Args:
        step_counts: sequence of number of leapfrog steps

    Returns:
        upper bound on number of leapfrog steps
    """
    ub_steps = 0
    for idx, step_count in enumerate(reversed(step_counts)):
        # Leapfrog integration with `step_count` number of steps is performed multiple
        # times when computing ghost samples.
        repetitions = 2**idx
        ub_steps += step_count * repetitions
    return ub_steps


@pytest.mark.parametrize("max_proposals", [1, 2, 3, 4, 5])
def test_drghmc_num_grad_evals_one_sample(max_proposals: int) -> None:
    model = StdNormal()
    model.log_density_gradient = _call_counter(model.log_density_gradient)  # type: ignore

    drghmc = DrGhmcDiag(
        model=model,
        max_proposals=max_proposals,
        leapfrog_step_sizes=[1.0] * max_proposals,
        leapfrog_step_counts=[2 * (i + 1) for i in range(max_proposals)],
        damping=0.001,
        prob_retry=True,
    )
    _ = drghmc.sample()

    ub_steps = upper_bound_leapfrog_steps(drghmc._leapfrog_step_counts)

    # Expect one call to log_density_gradient per leapfrog step, plus one for
    # calculating the density and gradient of the intial sample.
    assert model.log_density_gradient.calls <= 1 + ub_steps  # type: ignore


@pytest.mark.parametrize("num_samples", [1, 2, 3, 10])
def test_drghmc_num_grad_evals_many_samples(num_samples: int) -> None:
    model = StdNormal()
    model.log_density_gradient = _call_counter(model.log_density_gradient)  # type: ignore

    drghmc = DrGhmcDiag(
        model=model,
        max_proposals=1,
        leapfrog_step_sizes=[95.0],
        leapfrog_step_counts=[1],
        damping=0.001,
        prob_retry=True,
    )
    _ = np.array([drghmc.sample()[0] for _ in range(num_samples)])

    ub_steps_one_sample = upper_bound_leapfrog_steps(drghmc._leapfrog_step_counts)
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
        max_proposals=2,
        leapfrog_step_counts=[10, 10 * 2],
        leapfrog_step_sizes=[10.0, 0.5 / 5],
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
        max_proposals=3,
        leapfrog_step_counts=[10, 20, 30],
        leapfrog_step_sizes=[0.25, 0.25 / 4, 0.25 / 8],
        damping=0.2,
        init=init,
        seed=123,
    )

    drghmc_2 = DrGhmcDiag(
        model,
        max_proposals=3,
        leapfrog_step_counts=[10, 20, 30],
        leapfrog_step_sizes=[0.25, 0.25 / 4, 0.25 / 8],
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
    max_proposals = 3
    M = 800

    drghmc = DrGhmcDiag(
        model,
        max_proposals,
        leapfrog_step_sizes=[0.1] * max_proposals,
        leapfrog_step_counts=[3] * max_proposals,
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

    def drghmc_proposals(max_proposals: Any) -> DrGhmcDiag:
        drghmc = DrGhmcDiag(
            model=model,
            max_proposals=max_proposals,
            leapfrog_step_sizes=[1],
            leapfrog_step_counts=[1],
            damping=0.001,
            prob_retry=True,
        )
        return drghmc

    for non_integer_proposal in [[1], 1.0]:
        err_message = f"max_proposals must be an int, not {type(non_integer_proposal)}"
        with pytest.raises(TypeError, match=err_message):
            drghmc_proposals(non_integer_proposal)

    for non_positive_proposal in [0, -1]:
        err_message = (
            f"max_proposals must be greater than or equal to 1, not "
            f"{non_positive_proposal}"
        )
        with pytest.raises(ValueError, match=err_message):
            drghmc_proposals(non_positive_proposal)


def test_drghmc_invalid_leapfrog_step_sizes() -> None:
    model = StdNormal()

    def drghmc_step_sizes(step_sizes: Any, max_proposals: int) -> DrGhmcDiag:
        drghmc = DrGhmcDiag(
            model=model,
            max_proposals=max_proposals,
            leapfrog_step_sizes=step_sizes,
            leapfrog_step_counts=[1] * max_proposals,
            damping=0.001,
            prob_retry=True,
        )
        return drghmc

    max_proposals = 1
    for non_sequence_step_sizes in [1, 0.25]:
        err_message = (
            f"leapfrog_step_sizes must be an instance of type sequence, but found "
            f"type {type(non_sequence_step_sizes)}"
        )
        with pytest.raises(TypeError, match=err_message):
            drghmc_step_sizes(non_sequence_step_sizes, max_proposals)

    max_proposals = 2
    for incorrect_length_step_sizes in [[0.25], [0.25, 0.25, 0.25]]:
        err_message = (
            f"leapfrog_step_sizes must be a sequence of length {max_proposals}, so "
            f"that each proposal has its own specified leapfrog step size, but instead"
            f" found length of {len(incorrect_length_step_sizes)}"
        )
        with pytest.raises(ValueError, match=err_message):
            drghmc_step_sizes(incorrect_length_step_sizes, max_proposals)

    max_proposals = 2
    non_float_step_sizes = [float(1), int(1)]
    invalid_idx = 1
    err_message = (
        f"each step size in leapfrog_step_sizes must be of type float, but found step "
        f"size of type {type(non_float_step_sizes[invalid_idx])} at index "
        f"{invalid_idx}"
    )
    with pytest.raises(TypeError, match=err_message):
        drghmc_step_sizes(non_float_step_sizes, max_proposals)

    max_proposals = 2
    invalid_idx = 0
    for non_positive_step_sizes in [[-0.25, 0.25], [0.0, 0.25]]:
        err_message = (
            f"each step size in leapfrog_step_sizes must be positive, but found "
            f"step size of {non_positive_step_sizes[invalid_idx]} at index "
            f"{invalid_idx}"
        )
        with pytest.raises(ValueError, match=err_message):
            drghmc_step_sizes(non_positive_step_sizes, max_proposals)


def test_drghmc_invalid_leapfrog_step_counts() -> None:
    model = StdNormal()

    def drghmc_step_counts(step_counts: Any, max_proposals: int) -> DrGhmcDiag:
        drghmc = DrGhmcDiag(
            model=model,
            max_proposals=max_proposals,
            leapfrog_step_sizes=[1.0] * max_proposals,
            leapfrog_step_counts=step_counts,
            damping=0.001,
            prob_retry=True,
        )
        return drghmc

    max_proposals = 1
    for non_sequence_step_counts in [1, 0.25]:
        err_message = (
            f"leapfrog_step_counts must be an instance of type sequence, but found "
            f"type {type(non_sequence_step_counts)}"
        )
        with pytest.raises(TypeError, match=err_message):
            drghmc_step_counts(non_sequence_step_counts, max_proposals)

    max_proposals = 2
    for incorrect_length_step_counts in [[0.25], [0.25, 0.25, 0.25]]:
        err_message = (
            f"leapfrog_step_counts must be a sequence of length {max_proposals}, so "
            f"that each proposal has its own specified number of leapfrog steps, but "
            f"instead found length of {len(incorrect_length_step_counts)}"
        )
        with pytest.raises(ValueError, match=err_message):
            drghmc_step_counts(incorrect_length_step_counts, max_proposals)

    max_proposals = 2
    non_integer_step_counts = [1, 3.5]
    invalid_idx = 1
    err_message = (
        f"each step count in leapfrog_step_counts must be of type int, but found step "
        f"count of type {type(non_integer_step_counts[invalid_idx])} at index "
        f"{invalid_idx}"
    )
    with pytest.raises(TypeError, match=err_message):
        drghmc_step_counts(non_integer_step_counts, max_proposals)

    max_proposals = 2
    invalid_idx = 0
    for non_positive_step_counts in [[-2, 1], [0, 1]]:
        err_message = (
            f"each step count in leapfrog_step_counts must be positive, but found "
            f"step count of {non_positive_step_counts[invalid_idx]} at index "
            f"{invalid_idx}"
        )
        with pytest.raises(ValueError, match=err_message):
            drghmc_step_counts(non_positive_step_counts, max_proposals)


def test_drghmc_invalid_damping() -> None:
    model = StdNormal()

    def drghmc_proposals(damping: Any) -> DrGhmcDiag:
        max_proposals = 3
        drghmc = DrGhmcDiag(
            model=model,
            max_proposals=max_proposals,
            leapfrog_step_sizes=[1.0] * max_proposals,
            leapfrog_step_counts=[1] * max_proposals,
            damping=damping,
            prob_retry=True,
        )
        return drghmc

    for non_float_damping in [[0.5], int(1)]:
        err_message = (
            f"damping must be of type float, but found type {type(non_float_damping)}"
        )
        with pytest.raises(TypeError, match=err_message):
            drghmc_proposals(non_float_damping)

    for out_of_range_damping in [float(0), float(-1)]:
        err_message = (
            f"damping must be within (0, 1], but found damping of "
            f"{out_of_range_damping}"
        )
        with pytest.raises(ValueError, match=re.escape(err_message)):
            drghmc_proposals(out_of_range_damping)


def test_drghmc_iter() -> None:
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    drghmc = DrGhmcDiag(
        model,
        max_proposals=3,
        leapfrog_step_counts=[10, 10 * 2, 10 * 4],
        leapfrog_step_sizes=[0.25, 0.25 / 2, 0.25 / 4],
        damping=0.9,
        init=init,
    )

    M = 10000
    draws = np.array([draw for draw, _ in itertools.islice(drghmc, M)])

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)
