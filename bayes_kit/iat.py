import numpy as np
import numpy.typing as npt
import bayes_kit.autocorr as autocorr

FloatType = np.float64
IntType = np.int64
VectorType = npt.NDArray[FloatType]


def _end_pos_pairs(acor: VectorType) -> IntType:
    """
    Return the index 1 past the last positive pair of autocorrelations
    starting on an even index.  The sequence `acor` should contain
    autocorrelations from a Markov chain with values at the lag given by
    the index (i.e., `acor[0]` is autocorrelation at lag 0 and `acor[5]`
    is autocorrelation at lag 5).

    The even index pairs are (0, 1), (2, 3), (4, 5), ...  This function
    scans the pairs in order, and returns 1 plus the second index of the
    last such pair that has a positive sum.

    Examples:
    ```python
    _end_pos_pairs([]) = 0
    _end_pos_pairs([1]) = 0
    _end_pos_pairs([1, 0.4]) = 2
    _end_pos_pairs([1, -0.4]) = 2
    _end_pos_pairs([1, -0.5, 0.25, -0.3]) == 2
    _end_pos_pairs([1, -0.5, 0.25, -0.1]) == 4
    _end_pos_pairs([1, -0.5, 0.25, -0.3, 0.05]) == 2
    _end_pos_pairs([1, -0.5, 0.25, -0.1, 0.05]) == 4
    ```

    Parameters:
    acor (VectorType): Input sequence of autocorrelations at lag given by index.

    Returns:
    The index 1 past the last positive pair of values starting on an even index.
    """
    N = len(acor)
    n = 0
    while n + 1 < N:
        if acor[n] + acor[n + 1] < 0:
            return n
        n += 2
    return n


def iat_ipse(chain: VectorType) -> FloatType:
    """
    Return an estimate of the integrated autocorrelation time (IAT)
    of the specified Markov chain using the initial positive sequence
    estimator (IPSE).

    The integrated autocorrelation time of a chain is defined to be
    the sum of the autocorrelations at every lag (positive and negative).
    If `autocorr[n]` is the autocorrelation at lag `n`, then

    ```
    IAT = SUM_{n in Z} autocorr[n],
    ```

    where `Z = {..., -2, -1, 0, 1, 2, ...}` is the set of integers.

    Because the autocorrelations are symmetric, `autocorr[n] == autocorr[-n]` and
    `autocorr[0] = 1`, if we double count the non-negative entries, we will have
    counted `autocorr[0]`, which is 1, twice, so we subtract 1, to get

    ```
    IAT = -1 + 2 * SUM_{n in Nat} autocorr[n],
    ```

    where `Nat = {0, 1, 2, ...}` is the set of natural numbers.

    References:
        Geyer, Charles J. 2011. “Introduction to Markov Chain Monte Carlo.”
        In Handbook of Markov Chain Monte Carlo, edited by Steve Brooks,
        Andrew Gelman, Galin L. Jones, and Xiao-Li Meng, 3–48. Chapman;
        Hall/CRC.

    Parameters:
        chain: A Markov chain.

    Return:
        An estimate of the integrated autocorrelation time (IAT) for the specified chain.

    Raises:
    ValueError: if there are fewer than 4 elements in the chain
    """
    if len(chain) < 4:
        raise ValueError(f"ess requires len(chains) >= 4, but {len(chain)=}")
    acor = autocorr(chain)
    n = _end_pos_pairs(acor)
    return 2 * acor[0:n].sum() - 1


def iat_imse(chain: VectorType) -> FloatType:
    """
    Return an estimate of the integrated autocorrelation time (IAT)
    of the specified Markov chain using the initial monotone sequence
    estimator (IMSE).

    The IMSE imposes a monotonic downward condition on the sum of pairs,
    replacing each sum with the minimum of the sum and the minimum of
    the previous sums.

    References:
    Geyer, C.J., 1992. Practical Markov chain Monte Carlo. Statistical Science
    7(4):473--483.

    Geyer, Charles J. 2011. “Introduction to Markov Chain Monte Carlo.”
    In Handbook of Markov Chain Monte Carlo, edited by Steve Brooks,
    Andrew Gelman, Galin L. Jones, and Xiao-Li Meng, 3–48. Chapman;
    Hall/CRC.

    Parameters:
    chain: A Markov chain.

    Return:
    An estimate of integrated autocorrelation time (IAT) for the specified chain.

    Throws:
    ValueError: If there are fewer than 4 elements in the chain.
    """
    if len(chain) < 4:
        raise ValueError(f"iat requires len(chains) >=4, but {len(chain) = }")
    acor = autocorr(chain)
    n = _end_pos_pairs(acor)
    prev_min = acor[0] + acor[1]
    acor_sum = prev_min
    i = 2
    while i + 1 < n:
        # enforce monotone downward condition (slow loop)
        prev_min = min(prev_min, acor[i] + acor[i + 1])
        acor_sum += prev_min
        i += 2
    return 2 * acor_sum - 1


def iat(chain: VectorType) -> FloatType:
    """
    Return an estimate of the integrated autocorrelation time (IAT)
    of the specified Markov chain. Evaluated by delegating to the
    initial monotone sequence estimator, `iat_imse(chain)`.

    The IAT can be less than one in cases where the Markov chain is
    anti-correlated.

    Parameters:
    chain: A Markov chain.

    Return:
    The integrated autocorrelation time (IAT) for the specified chain.

    Throws:
    ValueError: If there are fewer than 4 elements in the chain.
    """
    return iat_imse(chain)
