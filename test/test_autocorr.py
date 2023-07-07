import numpy as np
import pytest as pt

from bayes_kit.autocorr import autocorr
from bayes_kit.typing import VectorType

from .test_iat import sample_ar1


def test_autocorr_fixed() -> None:
    y = np.asarray([1, 0, 0, 0])
    ac = autocorr(y)
    ac_expected = np.asarray([1.000, -0.083, -0.167, -0.250])
    np.testing.assert_allclose(ac_expected, ac, atol=0.001, rtol=0.001)


def autocorr_ar1(rho: float, N: int) -> VectorType:
    ac = np.zeros(N)
    ac[0] = 1
    for n in range(1, N):
        ac[n] = ac[n - 1] * rho
    return ac


def test_autocorr_ar1() -> None:
    N = 3000
    y = sample_ar1(0.5, N)
    ac = autocorr(y)
    ac_expected = autocorr_ar1(0.5, N)
    np.testing.assert_allclose(ac_expected[0:20], ac[0:20], atol=0.1, rtol=0.1)


def test_autocorr_independent() -> None:
    N = 3000
    y = np.random.normal(size=N)
    ac = autocorr(y)
    ac_expected = np.zeros(N)
    ac_expected[0] = 1
    np.testing.assert_allclose(ac_expected[0:20], ac[0:20], atol=0.1, rtol=0.1)


def test_autocorr_exceptions() -> None:
    with pt.raises(ValueError):
        autocorr([])
    with pt.raises(ValueError):
        autocorr([1.1])
    autocorr([1.1, 1.2])
