import numpy as np
import pytest as pt

import bayes_kit as bk

from .test_iat import integrated_autocorr_time_ar1, sample_ar1


def expected_ess_ar1(rho, N):
    return N / integrated_autocorr_time_ar1(rho)


def run_ess_test_ar1(rho, N):
    v = sample_ar1(rho, N)
    E_ess = expected_ess_ar1(rho, N)

    hat_ess1 = bk.ess(v)
    np.testing.assert_allclose(E_ess, hat_ess1, atol=N, rtol=0.1)

    hat_ess2 = bk.ess_imse(v)
    np.testing.assert_allclose(E_ess, hat_ess2, atol=N, rtol=0.1)

    hat_ess3 = bk.ess_ipse(v)
    np.testing.assert_allclose(E_ess, hat_ess3, atol=N, rtol=0.1)


def test_ess_ar1():
    # includes correlated (rho > 0), uncorrelated (rho = 0), and antianti-correlated tests (rho < 0)
    for rho in np.arange(-0.5, 0.5, step=0.2):
        run_ess_test_ar1(rho, 20_000)


def test_ess_independent():
    N = 10_000
    y = np.random.normal(size=N)
    hat_ess = bk.ess(y)
    E_ess = N
    np.testing.assert_allclose(E_ess, hat_ess, rtol=0.1)


def test_ess_exceptions():
    for n in range(4):
        v = sample_ar1(0.5, n)
        with pt.raises(ValueError):
            bk.ess(v)
        with pt.raises(ValueError):
            bk.ess_imse(v)
        with pt.raises(ValueError):
            bk.ess_ipse(v)
