import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy as sp

FloatType = np.float64
VectorType = NDArray[FloatType]
SeqType = ArrayLike


def split_chains(chains: list[SeqType]) -> list[SeqType]:
    """
    Return a list of the input chains split in half.  The result will be
    a list twice as long as the input.  For example,
    ```
    >>> split_chains([[1, 2, 3, 4], [5, 6, 7]])
    [[1, 2], [3, 4], [5, 6], [7]]
    ```

    Parameters:
    chains: List of univariate Markov chains.

    Returns:
    List of input chains split in half.
    """
    return [list(arr) for chain in chains for arr in np.array_split(chain, 2)]


def rank_chains(chains: list[SeqType]) -> list[SeqType]:
    """
    Returns a copy of the included Markov chains transformed with
    ranks normalized to [0, 1] and an offset inverse CDF. The ranks
    are ascending and start with 1 for the smallest value.

    Parameters:
    chains: list of univariate Markov chains

    Returns:
    List of chains with values replaced by transformed ranks.
    """
    if len(chains) == 0:
        return chains
    flattened = np.concatenate(chains)
    ranks = flattened.argsort().argsort() + 1
    reshaped_arrays = []
    current_index = 0
    for array in chains:
        size = len(array)
        reshaped_arrays.append(ranks[current_index : current_index + size])
        current_index += size
    return reshaped_arrays


def rank_normalize_chains(chains: list[SeqType]) -> list[SeqType]:
    """Return the rank-normalized version of the input chains.

    The rank-normalized value for element `j` of list `i` is
    ```
    inverse_Phi((rank[i][j] - 3/8) / (size(chains) - 1/4),
    ```
    where `inv_Phi` is the inverse cumulative distribution function for
    the standard normal distribution and
    ```
    rank[i][j] = rank_chains(chains)[i][j]
    ```
    is the rank of element `i` in chain `j` and `size(chains)` is the
    total number of elements in the chains.

    For a specification of ranking, see :func:`rank_chains`.

    Parameters:
    chains: List of univariate Markov chains.

    Returns:
    List of chains with values replaced by rank-normalized values.
    """
    S = sum([len(chain) for chain in chains])
    result = chains
    ranked_chains = rank_chains(chains)
    for i, chain_i in enumerate(ranked_chains):
        for j, rank_ij in enumerate(chain_i):
            val = sp.stats.norm.ppf((rank_ij - 0.325) / (S - 0.25))
            result[i][j] = val
    return result


def rhat(chains: list[SeqType]) -> FloatType:
    """
    Return the potential scale reduction factor (R-hat) for a list of Markov chains.

    If there are `M` chains of length `N[m]` each, with draws `theta[m, n]`,
    then `R-hat = sqrt((mean(N) - 1) / mean(N) + var(phi) / mean(psi))`, where
    `phi[m] = mean(chains[m])` and `psi[m] = var(chains[m])`.  This reduces to
    the standard definition when all chains are the same length.

    R-hat was introduced in the following paper.

    Gelman, A. and Rubin, D. B., 1992. Inference from iterative simulation using
    multiple sequences. Statistical Science, 457--472.

    Parameters:
    chains: list of univariate Markov chains

    Returns:
    R-hat statistic

    Throws:
    ValueError: If there is not at least one chain or if any chain has
    fewer than two elements.
    """
    if len(chains) < 2:
        raise ValueError(f"rhat requires len(chains) >= 2, but {len(chains) = }")
    if not all(len(chain) >= 2 for chain in chains):
        raise ValueError(f"rhat requires len(chain) >= 2 for all chain in chains")
    chain_lengths = [len(chain) for chain in chains]
    mean_chain_length = np.mean(chain_lengths)
    means = [np.mean(chain) for chain in chains]
    vars = [np.var(chain, ddof=1) for chain in chains]
    r_hat: np.float64 = np.sqrt(
        (mean_chain_length - 1) / mean_chain_length
        + np.var(means, ddof=1) / np.mean(vars)
    )
    return r_hat


def split_rhat(chains: list[SeqType]) -> FloatType:
    """
    Return the potential scale reduction factor (R-hat) for a list of
    Markov chains consisting of each of the input chains split in half.
    Unlike the base `rhat(chains)` function, this version is applicable
    to a single Markov chain.

    Split R-hat was introduced in the *Stan Reference Manual.*  The
    first official publication was in the following book.

    Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari,
    A. and Rubin, D.B., 2013. *Bayesian Data Analysis.* Third Edition.
    CRC press.

    See :func:`split_chains` for a specification of splitting.

    Parameters:
    chains: List of univariate Markov chains.

    Returns:
    Split R-hat statistic.

    Throws:
    ValueError: If there are fewer than two chains or if any chain has
    fewer than than four elements.
    """
    return rhat(split_chains(chains))


def rank_normalized_rhat(chains: list[SeqType]) -> FloatType:
    """Return the rank-normalized R-hat for the specified chains.
    Rank normalized r-hat replaces each value in the chains with its
    rank, applies a shifted inverse standard normal cdf, and
    returns the split R-hat value of the result.

    Rank-normalized R-hat was introduced in the following paper.

    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Bürkner,
    P.C., 2021. Rank-normalization, folding, and localization: An
    improved R-hat for assessing convergence of MCMC (with
    discussion). *Bayesian Analysis* 16(2):667-718.

    See :func:`split_rhat` for a specification of split R-hat and
    :func:`rank_normalize_chains` for rank normalization.

    Parameters:
    chains: List of univariate Markov chains.

    Returns:
    Rank-normalized R-hat statistic.

    Throws:
    ValueError: If there are fewer than two chains or if any chain has
    fewer than than four elements.
    """
    return split_rhat(rank_normalize_chains(chains))
