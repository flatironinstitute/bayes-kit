import numpy as np

FloatType = np.float64
VectorType = np.typing.NDArray[FloatType]


def rhat(chains: list[VectorType]) -> FloatType:
    """
    Return the potential scale reduction factor (R-hat) for a list of Markov chains.

    If there are `M` chains of length `N[m]` each, with draws `theta[m, n]`,
    then `R-hat = sqrt((mean(N) - 1) / mean(N) + var(phi) / mean(psi))`, where
    `phi[m] = mean(chains[m])` and `psi[m] = var(chains[m])`.  This reduces to
    the standard definition when all chains are the same length.

    Parameters:
    chains: list of univariate Markov chains

    Returns:
    R-hat statistic

    Throws:
    ValueError: if there are fewer than two chains
    """
    if len(chains) < 2:
        raise ValueError(f"rhat requires len(chains) >= 2, but {len(chains) = }")
    chain_lengths = [len(chain) for chain in chains]
    mean_chain_length = np.mean(chain_lengths)
    means = [np.mean(chain) for chain in chains]
    vars = [np.var(chain, ddof=1) for chain in chains]
    r_hat = np.sqrt(
        (mean_chain_length - 1) / mean_chain_length + np.var(means, ddof=1) / np.mean(vars)
    )
    return r_hat
