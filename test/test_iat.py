import numpy as np
import bayes_kit as bk
import pytest as pt
from bayes_kit.iat import _end_pos_pairs


def sample_ar1(rho, N):
    z = np.random.normal(size=N)
    for n in range(1, N):
        z[n] += rho * z[n - 1]
    return z


def integrated_autocorr_time_ar1(rho):
    last_sum = 0
    sum = 1
    t = 1
    while sum != last_sum:
        last_sum = sum
        sum += 2 * rho**t
        t += 1
    return sum


def run_iat_ar1_test(rho, N):
    v = sample_ar1(rho, N)
    E_iat = integrated_autocorr_time_ar1(rho)

    hat_iat1 = bk.iat(v)
    np.testing.assert_allclose(E_iat, hat_iat1, rtol=0.1)

    hat_iat2 = bk.iat_imse(v)
    np.testing.assert_allclose(E_iat, hat_iat2, rtol=0.1)

    hat_iat3 = bk.iat_ipse(v)
    np.testing.assert_allclose(E_iat, hat_iat3, rtol=0.1)


def test_iat_independent():
    E_iat = 1.0
    N = 20_000
    y = np.random.normal(size=N)
    hat_iat = bk.iat(y)
    np.testing.assert_allclose(E_iat, hat_iat, rtol=0.1)


def test_iat_ar1():
    # tests correlated (rho > 0) and anti-correlated (rho < 0) cases
    for rho in np.arange(-0.5, 0.5, step=0.2):
        run_iat_ar1_test(rho, 20_000)


def test_iat_exceptions():
    for n in range(4):
        v = sample_ar1(0.5, n)
        with pt.raises(ValueError):
            bk.iat(v)
        with pt.raises(ValueError):
            bk.iat_imse(v)
        with pt.raises(ValueError):
            bk.iat_ipse(v)


def pair_start_tester(chain, expected_pos):
    np.testing.assert_equal(expected_pos, _end_pos_pairs(chain))


def test_end_pos_pairs():
    pair_start_tester([], 0)
    pair_start_tester([1], 0)
    pair_start_tester([1, -0.5], 2)
    pair_start_tester([1, -0.5, 0.25], 2)
    pair_start_tester([1, -0.5, 0.25, -0.3], 2)
    pair_start_tester([1, -0.5, 0.25, -0.1], 4)
    pair_start_tester([1, -0.5, 0.25, -0.3, 0.05], 2)
    pair_start_tester([1, -0.5, 0.25, -0.1, 0.05], 4)
