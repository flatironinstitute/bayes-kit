import numpy as np
import numpy.typing as npt

FloatType = np.float64
IntType = int
VectorType = npt.NDArray[FloatType]

def autocorr_fft(chain: VectorType) -> VectorType:
    """Return the sample autocorrelations at all lags for the specified sequence.
    Algorithmically, this function calls a fast Fourier transform (FFT).

    Parameters:
        chain (VectorType): The sequence whose autocorrelation is returned.

    Returns:
        Autocorrelation estimates at all lags for the specified sequence.
    """
    size = 2 ** np.ceil(np.log2(2 * len(chain) - 1)).astype("int")
    var = np.var(chain)
    ndata = chain - np.mean(chain)
    fft = np.fft.fft(ndata, size)
    pwr = np.abs(fft) ** 2
    N = len(ndata)
    acorr: VectorType = np.fft.ifft(pwr).real / var / N
    return acorr

def autocorr_np(chain: VectorType) -> VectorType:
    """Return sample autocorrelations at all lags for the specified sequence.
    Algorithmically, this function delegates to the NumPy `correlation()` function.

    Parameters:
        chain (VectorType): sequence whose autocorrelation is returned

    Returns:
        The autocorrelation estimates at all lags for the specified sequence.
    """
    chain_ctr = chain - np.mean(chain)
    N = len(chain_ctr)
    acorr: VectorType = np.correlate(chain_ctr, chain_ctr, "full")[N - 1 :] / N
    return acorr

def autocorr(chain: VectorType) -> VectorType:
    """
    Return sample autocorrelations at all lags for the specified sequence.
    Algorithmically, this function delegates to `autocorr_fft`.

    Parameters:
    chain: sequence whose autocorrelation is returned

    Returns:
    autocorrelation estimates at all lags for the specified sequence
    """
    return autocorr_fft(chain)

def first_neg_pair_start(chain: VectorType) -> IntType:
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
        n = n + 2
    return N

def ess_ipse(chain: VectorType) -> float:
    """
    Return an estimate of the effective sample size (ESS) of the specified Markov chain
    using the initial positive sequence estimator (IPSE).

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
    n = first_neg_pair_start(acor)
    sigma_sq_hat = acor[0] + 2 * acor[1:n].sum()
    ess : float = len(chain) / sigma_sq_hat
    return ess

def ess_imse(chain: VectorType) -> float:
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
    n = first_neg_pair_start(acor)
    prev_min = acor[1] + acor[2]
    # convex minorization uses slow loop
    accum = prev_min
    i = 3
    while i + 1 < n:
        minprev = min(prev_min, acor[i] + acor[i + 1])
        accum = accum + minprev
        i = i + 2
    # end diff code
    sigma_sq_hat = acor[0] + 2 * accum
    ess: float = len(chain) / sigma_sq_hat
    return ess

def ess(chain: VectorType) -> float:
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
    

