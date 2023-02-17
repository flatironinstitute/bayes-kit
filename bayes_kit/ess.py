import numpy as np
import numpy.typing as npt
import bayes_kit.autocorr as autocorr

FloatType = np.float64
IntType = np.int64
VectorType = npt.NDArray[FloatType]

def _first_even_neg_pair_start(chain: VectorType) -> IntType:
    """
    Return the index of first element of the sequence whose sum with the following
    element is negative, or the length of the sequence if there is no such element.
    
    Parameters:
    chain: input sequence

    Return:
    index of first element whose sum with following element is negative, or
    the number of elements if there is no such element
    """
    N = len(chain)
    n = 0
    while n + 1 < N:
        if chain[n] + chain[n + 1] < 0:
            return n
        n += 2
    return N

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
    n = _first_even_neg_pair_start(acor)
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
    n = _first_even_neg_pair_start(acor)
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
    

