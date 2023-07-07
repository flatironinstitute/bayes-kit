import numpy as np
from typing import Union, Sequence
from numpy.typing import NDArray
import scipy as sp

FloatType = np.float64
VectorType = NDArray[FloatType]
SeqType = Union[Sequence[float], NDArray[np.float64]]

def split_chains(chains: list[SeqType]) -> list[SeqType]:
    """Return a list of the input chains split in half.  The result will
    be a list twice as long as the input.  For odd sized chains, the
    first half will be one element longer.  For example,
    ```
    >>> split_chains([[1, 2, 3], [4, 5, 6, 7]])
    [[1, 2], [3], [4, 5], [6, 7]]
    ```

    Args:
    chains: List of univariate Markov chains.

    Returns:
    List of input chains split in half.
    """
    return [arr for chain in chains for arr in np.array_split(chain, 2)]

def rank_chains(chains: list[SeqType]) -> list[SeqType]:
    """
    Returns a copy of the included Markov chains with all values
    transformed to ranks.  Ranks are ascending and start at 1.

    For example,
    ```python
    >>> rank_chains([[4.2, 5.7], [7.2, 6.1], [-12.9, 107]])
    [[2, 3], [5, 4], [1, 6]]
    ```
    The values in the chains and the ranks are
    ```
    Values: -12.9, 4.2, 5.7, 6.1, 7.2, 107
    Ranks:      1,   2,   3    4    5,   6
    ```

    Args:
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

    Rank normalization maps the ranks to the range (0, 1) and then returns
    the quantiles of the standard-normal distribution for the resulting
    values. A small margin is first applied to avoid infinite values from
    the ppf function.

    The rank-normalized value for element `j` of list `i` is
    ```
    inv_Phi((rank[i][j] - 3/8) / (size(chains) - 1/4),
    ```
    where 
    * `inv_Phi` is the inverse cumulative distribution function for
    the standard normal distribution,
    * `rank[i][j] = rank_chains(chains)[i][j]` is the rank of element
    `i` in chain `j`, and 
    * `size(chains)` is the total number of elements in the chains.

    For a specification of ranking, see :func:`rank_chains`.

    The transformed values will be in the same order as the original
    values,
    ```python
    >>> rank_normalize_chains([[4.2, 5.7], [7.2, 6.1], [-12.9, 107]])
    [[-0.550, -0.087], [0.889, 0.356], [-1.188, 2.225]]
    ```

    The specific transform used, with constants 3/8 and 1/4, was
    introduced in the following book.

    Blom, G. (1958). Statistical Estimates and Transformed
    Beta-Variables. Wiley; New York.

    Args:
    chains: List of univariate Markov chains.

    Returns:
    List of chains with values replaced by rank-normalized values.
    """
    S = sum([len(chain) for chain in chains])
    result = []
    for chain_i in rank_chains(chains):
        result.append([sp.stats.norm.ppf((rank_ij - 0.325) / (S - 0.25)) for rank_ij in chain_i])
    return result

def rhat(chains: list[SeqType]) -> FloatType:
    """Return the potential scale reduction factor (R-hat) for a list of
    Markov chains.

    The R-hat value indicates how much the scale (i.e., standard
    deviation) of the distribution of values in the chains might be
    reduced by running longer.  If all chains have converged to an
    equilibrium distribution, the value of R-hat will be 1; if they have
    not converged, R-hat will be greater than 1.  As chain length
    increases, R-hat will converge to 1 if the Markov chains are well
    behaved in the sense of having the correct stationary distribution.

    Suppose there are `M` chains of length `N[m]` each, with draws
    `chains[m, n]`.  In particular, note that `N`, `phi`, and `psi` are
    all arrays.  Define the R-hat statistic as
    ```
    R-hat = sqrt((mean(N) - 1) / mean(N) + var(phi) / mean(psi)),
    ```
    where
    ```
    phi[m] = mean(chains[m])
    ```
    is the sample mean of chain `m` (i.e., `np.mean(chains[m])` in
    NumPy) and
    ```
    psi[m] = var(chains[m])
    ```
    is the sample variance of chain `m` (i.e., `np.var(chains[m],
    ddof=1)` in NumPy).

    R-hat was introduced in the following paper.

    Gelman, A. and Rubin, D. B., 1992. Inference from iterative simulation using
    multiple sequences. Statistical Science, 457--472.

    This function reduces to the definition in the paper when all the
    chains are the same length.

    Args:
    chains: List of univariate Markov chains.

    Returns:
    R-hat statistic.

    Throws:
    ValueError: If there is not at least one chain.
    ValueError: If any chain has fewer than two elements.
    """
    if len(chains) < 2:
        raise ValueError(f"rhat requires len(chains) >= 2, but {len(chains) = }")
    if not all(len(chain) >= 2 for chain in chains):
        raise ValueError(f"rhat requires len(chain) >= 2 for every chain in chains")
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
    """Return the potential scale reduction factor (R-hat) for a list of
    Markov chains consisting of each of the input chains split in half.

    The main utility of splitting is to diagnose non-stationary chains
    (e.g., ones with an upward or downward trend).  Unlike the base
    `rhat(chains)` function, this version is applicable to a single
    Markov chain.

    Split R-hat was introduced in the *Stan Reference Manual.*  The
    first official publication was in the following book.

    Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari,
    A. and Rubin, D.B., 2013. *Bayesian Data Analysis.* Third Edition.
    CRC press.

    See :func:`split_chains` for a definition of splitting.

    Args:
    chains: List of univariate Markov chains.

    Returns:
    Split R-hat statistic.

    Throws:
    ValueError: If there are no chains.
    ValueError: If any chain has fewer than than four elements.
    """
    return rhat(split_chains(chains))

def rank_normalized_rhat(chains: list[SeqType]) -> FloatType:
    """Return the rank-normalized R-hat for the specified chains.
    Rank normalized r-hat replaces each value in the chains with its
    rank, applies a shifted inverse standard normal cdf, and
    returns the split R-hat value of the result.

    Rank-normalized R-hat should be more robust in situations where the
    stationary distribution of the Markov chains is not normal (e.g.,
    for density targets for which means and/or variances are undefined,
    such as the Cauchy distribution).

    Rank-normalized R-hat was introduced in the following paper.

    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Bürkner,
    P.C., 2021. Rank-normalization, folding, and localization: An
    improved R-hat for assessing convergence of MCMC (with
    discussion). *Bayesian Analysis* 16(2):667-718.

    See :func:`split_rhat` for a specification of split R-hat and
    :func:`rank_normalize_chains` for rank normalization.

    Args:
    chains: List of univariate Markov chains.

    Returns:
    Rank-normalized R-hat statistic.

    Throws:
    ValueError: If there are fewer than two chains.
    ValueError: If any chain has fewer than than four elements.
    """
    return split_rhat(rank_normalize_chains(chains))
