import numpy as np
import pytest as pt

import bayes_kit as bk
from bayes_kit.autocorr import autocorr


def test_autocorr_fixed():
    y = np.asarray([1, 0, 0, 0])
    ac = autocorr(y)
    ac_expected = np.asarray([1.000, -0.083, -0.167, -0.250])
    np.testing.assert_allclose(ac_expected, ac, atol=0.001, rtol=0.001)


def sample_ar1(rho, N):
    z = np.random.normal(size=N)
    for n in range(1, N):
        z[n] += rho * z[n - 1]
    return z


def autocorr_ar1(rho, N):
    ac = np.zeros(N)
    ac[0] = 1
    for n in range(1, N):
        ac[n] = ac[n - 1] * rho
    return ac


def test_autocorr_ar1():
    N = 3000
    y = sample_ar1(0.5, N)
    ac = autocorr(y)
    ac_expected = autocorr_ar1(0.5, N)
    np.testing.assert_allclose(ac_expected[0:20], ac[0:20], atol=0.1, rtol=0.1)


def test_autocorr_independent():
    N = 3000
    y = np.random.normal(size=N)
    ac = autocorr(y)
    ac_expected = np.zeros(N)
    ac_expected[0] = 1
    np.testing.assert_allclose(ac_expected[0:20], ac[0:20], atol=0.1, rtol=0.1)


def test_autocorr_exceptions():
    with pt.raises(ValueError):
        bk.autocorr([])
    with pt.raises(ValueError):
        bk.autocorr([1.1])
    bk.autocorr([1.1, 1.2])
