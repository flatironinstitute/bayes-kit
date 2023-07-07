from typing import List

import numpy as np
import numpy.typing as npt
import pytest as pt

import bayes_kit as bk
from bayes_kit.iat import _end_pos_pairs

VectorType = npt.NDArray[np.float64]


def sample_ar1(rho: float, N: int) -> VectorType:
    z = np.random.normal(size=N)
    for n in range(1, N):
        z[n] += rho * z[n - 1]
    return z


def integrated_autocorr_time_ar1(rho: float) -> float:
    last_sum = 0.0
    sum = 1.0
    t = 1
    while sum != last_sum:
        last_sum = sum
        sum += 2 * rho**t
        t += 1
    return sum


def run_iat_ar1_test(rho: float, N: int) -> None:
    v = sample_ar1(rho, N)
    E_iat = integrated_autocorr_time_ar1(rho)

    hat_iat1 = bk.iat(v)
    np.testing.assert_allclose(E_iat, hat_iat1, rtol=0.1)

    hat_iat2 = bk.iat_imse(v)
    np.testing.assert_allclose(E_iat, hat_iat2, rtol=0.1)

    hat_iat3 = bk.iat_ipse(v)
    np.testing.assert_allclose(E_iat, hat_iat3, rtol=0.1)


def test_iat_independent() -> None:
    E_iat = 1.0
    N = 20_000
    y = np.random.normal(size=N)
    hat_iat = bk.iat(y)
    np.testing.assert_allclose(E_iat, hat_iat, rtol=0.1)


def test_iat_ar1() -> None:
    # tests correlated (rho > 0) and anti-correlated (rho < 0) cases
    for rho in np.arange(-0.5, 0.5, step=0.2):
        run_iat_ar1_test(rho, 20_000)


def test_iat_exceptions() -> None:
    for n in range(4):
        v = sample_ar1(0.5, n)
        with pt.raises(ValueError):
            bk.iat(v)
        with pt.raises(ValueError):
            bk.iat_imse(v)
        with pt.raises(ValueError):
            bk.iat_ipse(v)


def pair_start_tester(chain: List[float], expected_pos: int) -> None:
    np.testing.assert_equal(expected_pos, _end_pos_pairs(chain))


def test_end_pos_pairs() -> None:
    pair_start_tester([], 0)
    pair_start_tester([1], 0)
    pair_start_tester([1, -0.5], 2)
    pair_start_tester([1, -0.5, 0.25], 2)
    pair_start_tester([1, -0.5, 0.25, -0.3], 2)
    pair_start_tester([1, -0.5, 0.25, -0.1], 4)
    pair_start_tester([1, -0.5, 0.25, -0.3, 0.05], 2)
    pair_start_tester([1, -0.5, 0.25, -0.1, 0.05], 4)
