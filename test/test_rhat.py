import numpy as np
import scipy as sp
from bayes_kit.rhat import (
    rhat,
    split_chains,
    split_rhat,
    rank_chains,
    rank_normalize_chains,
    rank_normalized_rhat,
)
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
    # test API implementation vs. brute-force implementation rhat_expected
    chain1 = [1.01, 1.05, 0.98, 0.90, 1.23]
    chain2 = [0.99, 1.00, 1.01, 1.15, 0.83]
    chain3 = [0.84, 0.90, 0.94, 1.10, 0.92]
    chain4 = [0.32, 1.81, 0.90, 0.10, 2.85]
    chains = [chain1, chain2]
    np.testing.assert_allclose(rhat_expected(chains), rhat(chains), atol=0.1, rtol=0.2)
    chains = [chain1, chain2, chain3]
    np.testing.assert_allclose(rhat_expected(chains), rhat(chains), atol=0.1, rtol=0.2)
    chains = [chain1, chain2, chain3, chain4]
    np.testing.assert_allclose(rhat_expected(chains), rhat(chains), atol=0.1, rtol=0.2)

def test_rhat_ragged():
    chain1 = [1.01, 1.05, 0.98, 0.90, 1.23]
    chain2 = [0.99, 1.00, 1.01, 1.15, 0.83, 0.95]
    chains = [chain1, chain2]
    rhat_est = rhat(chains)
    np.testing.assert_allclose(rhat_expected(chains), rhat(chains), atol=0.1, rtol=0.2)

def rhat_throws(chains):
    with pt.raises(ValueError):
        rhat(chains)
    
def test_rhat_at_least_two_chains():
    rhat_throws([])
    rhat_throws([[1.01, 1.2, 1.3, 1.4]],)

def test_rhat_at_least_two_elements_per_chain():
    rhat_throws([[1, 2, 3], [4], [5, 6, 7, 8, 9]])

def test_split_chains():
    np.testing.assert_equal([], split_chains([]))
    np.testing.assert_equal([[1], []], split_chains([[1]]))
    np.testing.assert_equal([[1], [2]], split_chains([[1, 2]]))
    np.testing.assert_equal([[1, 2], [3]], split_chains([[1, 2, 3]]))
    np.testing.assert_equal([[1, 2, 3], [4, 5, 6]], split_chains([[1, 2, 3, 4, 5, 6]]))
    np.testing.assert_equal(
        [[1, 2], [3], [4, 5], [6, 7]], split_chains([[1, 2, 3], [4, 5, 6, 7]])
    )

def test_split_rhat():
    # split_rhat should return same result as rhat on split chains
    np.testing.assert_allclose(rhat([[1, 2], [3, 4]]), split_rhat([[1, 2, 3, 4]]))
    np.testing.assert_allclose(
        rhat([[1, 2, 2], [3, 4, 3]]), split_rhat([[1, 2, 2, 3, 4, 3]])
    )
    np.testing.assert_allclose(
        rhat([[1, -2, 3], [4, 5, 6], [7, 8], [9, 12]]),
        split_rhat([[1, -2, 3, 4, 5, 6], [7, 8, 9, 12]]),
    )

def split_rhat_throws(chains):
    with pt.raises(ValueError):
        split_rhat(chains)
        
def test_split_rhat_at_least_one_chain():
    split_rhat_throws([])

def test_split_rhat_at_least_four_elements_per_chain():
    split_rhat_throws([[1, 2, 3]])
    split_rhat_throws([[1, 2, 3, 4], [1, 2, 3]])
    split_rhat_throws([[1, 2, 3], [1, 2, 3, 4]])

def rank_chains_equal(ranks, chains):
    np.testing.assert_equal(ranks, rank_chains(chains))
    
def test_rank_chains():
    rank_chains_equal([], [])
    rank_chains_equal([[1]], [[2.3]])
    rank_chains_equal([[1, 2]], [[2.3, 4.9]])
    rank_chains_equal([[2, 3, 1]], [[3.9, 5.2, 2.1]])
    rank_chains_equal([[2], [1]], [[4.2], [1.9]])
    rank_chains_equal([[2, 3], [1, 4]], [[4.2, 5.7], [1.9, 12.2]])
    rank_chains_equal(
        [[2, 3], [5, 4], [1, 6]],
        [[4.2, 5.7], [7.2, 6.1], [-12.9, 107]]
    )

def rank_norm(r, S):
    return sp.stats.norm.ppf((r - 0.325) / (S - 0.25))

def rank_normalize_chains_close(ranks, chains):
    np.testing.assert_equal(ranks, rank_normalize_chains(chains))
    
def test_rank_normalize_chains():
    rank_normalize_chains_close([], [])

    rn11 = rank_norm(1, 1)
    rank_normalize_chains_close([[rn11]], [[32.7]])

    rn12 = rank_norm(1, 2)
    rn22 = rank_norm(2, 2)
    rank_normalize_chains_close([[rn22, rn12]], [[3.9, 1.8]])

    rn13 = rank_norm(1, 3)
    rn23 = rank_norm(2, 3)
    rn33 = rank_norm(3, 3)
    rank_normalize_chains_close(
        [[rn23, rn33, rn13]], [[3.5, 5.9, 1.0]]
    )

    rn14 = rank_norm(1, 4)
    rn24 = rank_norm(2, 4)
    rn34 = rank_norm(3, 4)
    rn44 = rank_norm(4, 4)
    rank_normalize_chains_close(
        [[rn34, rn24], [rn14, rn44]], [[3.9, 3.1], [2.2, 5.9]]
    )

def rank_normalized_rhat_throws(chains):
    with pt.raises(ValueError):
        rank_normalized_rhat(chains)

def test_rank_normalized_rhat_at_least_one_chain():
    rank_normalized_rhat_throws([])

def test_rank_normalized_rhat_at_least_four_elemenets_per_chain():
    rank_normalized_rhat_throws([[1.01, 1.2, 1.3]])
    rank_normalized_rhat_throws([[1, 2, 3], [4]])

def rank_normalized_rhat_close(ranks, chains):
    np.testing.assert_allclose(
        split_rhat(ranks), rank_normalized_rhat(chains)
    )

def test_rank_normalized_rhat():
    # expect rank-normalized-rhat to be equivalent to split_rhat on ranks
    rn14 = rank_norm(1, 4)
    rn24 = rank_norm(2, 4)
    rn34 = rank_norm(3, 4)
    rn44 = rank_norm(4, 4)
    rank_normalized_rhat_close(
        [[rn14, rn44, rn34, rn24]],
        [[1.8, 10.9, 6.3, 5.1]]
    )

    rn18 = rank_norm(1, 8)
    rn28 = rank_norm(2, 8)
    rn38 = rank_norm(3, 8)
    rn48 = rank_norm(4, 8)
    rn58 = rank_norm(5, 8)
    rn68 = rank_norm(6, 8)
    rn78 = rank_norm(7, 8)
    rn88 = rank_norm(8, 8)
    rank_normalized_rhat_close(
        [[rn28, rn38, rn78, rn88], [rn18, rn48, rn68, rn58]],
        [[2, 3, 7, 8], [1, 4, 6, 5]]
    )
