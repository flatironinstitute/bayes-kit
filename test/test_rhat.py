import numpy as np
from bayes_kit.rhat import rhat
import pytest as pt

def rhat_expected(chains):
    # uses brute force definition from BDA3
    psij_bar = [np.mean(c) for c in chains]
    psi_bar = np.mean(psij_bar)
    M = len(chains)
    N = len(chains[0])
    B = N / (M - 1) * sum((psij_bar - psi_bar) ** 2)
    s_sq = [1 / (N - 1) * sum((chains[m] - psij_bar[m]) ** 2) for m in range(M)]
    W = 1 / M * sum(s_sq)
    var_plus = (N - 1) / N * W + 1 / N * B
    rhat = np.sqrt(var_plus / W)
    return rhat

def test_rhat():
    chain1 = [1.01, 1.05, 0.98, 0.90, 1.23]
    chain2 = [0.99, 1.00, 1.01, 1.15, 0.83]
    chain3 = [0.84, 0.90, 0.94, 1.10, 0.92]
    chains = [chain1, chain2, chain3]
    np.testing.assert_allclose(rhat_expected(chains), rhat(chains), atol=0.1, rtol=0.2)

def test_rhat_ragged():
    chain1 = [1.01, 1.05, .98, .90, 1.23]
    chain2 = [0.99, 1.00, 1.01, 1.15, 0.83, 0.95]
    chains = [chain1, chain2]
    rhat_est = rhat(chains)
    np.testing.assert_allclose(rhat_expected(chains), rhat(chains), atol=0.1, rtol=0.2)

def test_rhat_exceptions():
    with pt.raises(ValueError):
        chains = []
        rhat(chains)
    with pt.raises(ValueError):
        chain1 = [1.01, 1.2, 1.3, 1.4]
        chains = [chain1]
        rhat(chains)

