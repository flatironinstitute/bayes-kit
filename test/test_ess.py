import numpy as np
import bayes_kit as bk
from bayes_kit.ess import ess
from bayes_kit.ess import ess_ipse
from bayes_kit.ess import ess_imse
import pytest as pt

def sample_ar1(rho, N):
    z = np.random.normal(size = N)
    for n in range(1, N):
        z[n] += rho * z[n - 1]
    return z

def integrated_autocorr_time(rho):
    sum = 0
    for t in range(1, 500):
        sum += rho**t
    return 1 + 2 * sum

def expected_ess(rho, N):
    return N / integrated_autocorr_time(rho)
    
def run_ess_test(rho, N):
    v = sample_ar1(rho, N)
    E_ess = expected_ess(rho, N)
    hat_ess1 = ess(v)
    print(f"{E_ess = }  {hat_ess1 = }")
    np.testing.assert_allclose(E_ess, hat_ess1, atol=N, rtol=0.1)
    hat_ess2 = ess_imse(v)
    np.testing.assert_allclose(E_ess, hat_ess2, atol=N, rtol=0.1)
    hat_ess3 = ess_ipse(v)
    np.testing.assert_allclose(E_ess, hat_ess3, atol=N, rtol=0.1)

def test_ess():
    for rho in np.arange(-0.1, 0.6, step = 0.1):
        run_ess_test(rho / 10, 100_000)

def test_ess_exceptions():
    for n in range(4):
        v = sample_ar1(0.5, n)
        with pt.raises(ValueError):
            ess(v)
        with pt.raises(ValueError):
            ess_imse(v)
        with pt.raises(ValueError):
            ess_ipse(v)
