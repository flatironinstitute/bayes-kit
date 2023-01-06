import numpy as np
import numpy.typing as npt

FloatType = np.float64
VectorType = npt.NDArray[FloatType]


def autocorr_fft(chain: VectorType) -> FloatType:
    size = 2 ** np.ceil(np.log2(2 * len(chain) - 1)).astype("int")
    var = np.var(chain)
    ndata = chain - np.mean(chain)
    fft = np.fft.fft(ndata, size)
    pwr = np.abs(fft) ** 2
    N = len(ndata)
    acorr = np.fft.ifft(pwr).real / var / N
    return acorr


def autocorr(chain: VectorType) -> FloatType:
    chain_ctr = chain - np.mean(chain)
    N = len(chain_ctr)
    acorrN = np.correlate(chain_ctr, chain_ctr, "full")[N - 1 :]
    return acorrN / N


def first_neg_pair_start(chain: VectorType):
    N = len(chain)
    n = 1
    while n + 1 < N:
        if chain[n] + chain[n + 1] < 0:
            return n
        n = n + 2
    return N


def ess(chain: VectorType):
    """initial positive sequence estimator (IPSE)"""
    if len(chain) < 4:
        raise ValueError(f"ess requires len(chains) >=4, but {len(chain) = }")
    acor = autocorr(chain)
    n = first_neg_pair_start(acor)
    sigma_sq_hat = acor[0] + 2 * sum(acor[1:n])
    ess = len(chain) / sigma_sq_hat
    return ess


def ess_mono(chain: VectorType):
    """initial monotone sequence estimator (IMSE)"""
    if len(chain) < 4:
        raise ValueError(f"ess requires len(chains) >=4, but {len(chain) = }")
    acor = autocorr(chain)
    n = first_neg_pair_start(acor)
    prev_min = acor[1] + acor[2]
    # convex minorization here much slower
    accum = prev_min
    i = 3
    while i + 1 < n:
        minprev = min(prev_min, acor[i] + acor[i + 1])
        accum = accum + minprev
        i = i + 2
    # end diff code
    sigma_sq_hat = acor[0] + 2 * accum
    ess = len(chain) / sigma_sq_hat
    return ess
