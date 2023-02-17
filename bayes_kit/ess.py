import numpy as np
import numpy.typing as npt
import bayes_kit.autocorr as autocorr

FloatType = np.float64
IntType = np.int64
VectorType = npt.NDArray[FloatType]


def _end_pos_pairs(chain: VectorType) -> IntType:
    """Return the index that is 1 past the last positive pair of values
    starting on an even index. The even index pairs are (0, 1), (2, 3),
    (4, 5), ...  The algorithm looks at the pairs in order, and returns
    1 plus the second index of the last such pair that has a positive
    sum.

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
        chain: Input sequence (typically of autocorrelations).

    Return:
        The index 1 past the last positive pair of values starting on an even index
    """
    N = len(chain)
    n = 0
    while n + 1 < N:
        if chain[n] + chain[n + 1] < 0:
            return n
        n += 2
    return n


def ess_ipse(chain: VectorType) -> FloatType:
    """
    Return an estimate of the effective sample size (ESS) of the specified Markov chain
    using the initial positive sequence estimator (IPSE).

    Parameters:
        chain: Markov chain whose ESS is returned

    Return:
        estimated effective sample size for the specified Markov chain

    Raises:
        ValueError: if there are fewer than 4 elements in the chain
    """
    if len(chain) < 4:
        raise ValueError(f"ess requires len(chains) >= 4, but {len(chain)=}")
    acor = autocorr(chain)
    n = _end_pos_pairs(acor)
    iat = 2 * acor[0:n].sum() - 1
    ess = len(chain) / iat
    return ess


def ess_imse(chain: VectorType) -> FloatType:
    """
    Return an estimate of the effective sample size (ESS) of the specified Markov chain
    using the initial monotone sequence estimator (IMSE).  This is the most accurate
    of the available ESS estimators.  Because of the convex minorization used,
    this approach is slower than using the IPSE function `ess_ipse`.

    This estimator was introduced in the following paper.

    Geyer, C.J., 1992. Practical Markov chain Monte Carlo. Statistical Science
    7(4):473--483.

    Parameters:
        chain: Markov chain whose ESS is returned

    Return:
        estimated effective sample size for the specified Markov chain

    Throws:
        ValueError: if there are fewer than 4 elements in the chain
    """
    if len(chain) < 4:
        raise ValueError(f"ess requires len(chains) >=4, but {len(chain) = }")
    acor = autocorr(chain)
    n = _end_pos_pairs(acor)
    prev_min = acor[0] + acor[1]
    # convex minorization to enforce monotonic downward estimates uses slow loop
    accum = prev_min
    i = 2
    while i + 1 < n:
        prev_min = min(prev_min, acor[i] + acor[i + 1])
        accum = accum + prev_min
        i += 2
    # end diff code
    iat = 2 * accum - 1
    ess = len(chain) / iat
    return ess


def ess(chain: VectorType) -> FloatType:
    """
    Return an estimate of the effective sample size of the specified Markov chain
    using the default ESS estimator (currently IMSE).  Evaluated by delegating
    to `ess_imse()`.

    Parameters:
        chain: Markov chains whose ESS is returned

    Return:
        estimated effective sample size for the specified Markov chain

    Throws:
        ValueError: if there are fewer than 4 elements in the chain
    """
    return ess_imse(chain)
