import numpy as np
from numpy.typing import NDArray, ArrayLike

FloatType = np.float64
VectorType = NDArray[FloatType]
SeqType = ArrayLike


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
    ValueError: if there are fewer than two chains
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


def split_chains(chains: list[SeqType]) -> list[SeqType]:
    """
    Return a list of the input chains split in half.  The result will be
    a list twice as long as the input.  For example, given
    ```
    >>> split_chains([[1,2,3,4],[5,6, 7]])
    [[1, 2], [3, 4], [5, 6], [7]]
    ```

    Parameters:
    chains: list of univariate Markov chains

    Returns:
    List of input chains split in half.
    """
    return [list(arr) for chain in chains for arr in np.array_split(chain, 2)]


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

    Parameters:
    chains: list of univariate Markov chains

    Returns:
    Split R-hat statistic
    """
    return rhat(split_chains(chains))

