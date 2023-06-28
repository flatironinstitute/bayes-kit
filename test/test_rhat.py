import numpy as np
from bayes_kit.rhat import rhat, split_chains, split_rhat
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
    chain1 = [1.01, 1.05, 0.98, 0.90, 1.23]
    chain2 = [0.99, 1.00, 1.01, 1.15, 0.83, 0.95]
    chains = [chain1, chain2]
    rhat_est = rhat(chains)
    np.testing.assert_allclose(rhat_expected(chains), rhat(chains),
                                   atol=0.1, rtol=0.2)

def test_rhat_size_exceptions():
    bad1 = []
    bad2 = [[1.01, 1.2, 1.3, 1.4]],
    bad3 = [[1, 2, 3], [4]]
    for chains in [bad1, bad2, bad3]:
        with pt.raises(ValueError):
            rhat(chains)

def test_split_chains():
    print("hello")
    np.testing.assert_equal([], split_chains([]))
    np.testing.assert_equal([[1], []], split_chains([[1]]))
    np.testing.assert_equal([[1], [2]], split_chains([[1, 2]]))
    np.testing.assert_equal([[1, 2], [3]], split_chains([[1, 2, 3]]))
    np.testing.assert_equal([[1, 2, 3], [4, 5, 6]],
                                split_chains([[1, 2, 3, 4, 5, 6]]))
    np.testing.assert_equal([[1, 2], [3], [4, 5], [6, 7]],
                                split_chains([[1, 2, 3], [4, 5, 6, 7]]))

def test_split_rhat():
    np.testing.assert_allclose(rhat([[1, 2], [3, 4]]),
                                   split_rhat([[1, 2, 3, 4]]))
    np.testing.assert_allclose(rhat([[1, 2, 2], [3, 4, 3]]),
                                   split_rhat([[1, 2, 2, 3, 4, 3]]))
    np.testing.assert_allclose(rhat([[1, -2, 3], [4, 5, 6], [7, 8], [9, 12]]),
                                   split_rhat([[1, -2, 3, 4, 5, 6], [7, 8, 9, 12]]))
                            
