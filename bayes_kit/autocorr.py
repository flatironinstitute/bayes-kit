import numpy as np
import numpy.typing as npt
from typing import Union, Sequence

FloatType = np.float64
VectorType = npt.NDArray[FloatType]
ChainType = Union[Sequence[float], VectorType]


def autocorr(chain: ChainType) -> VectorType:
    """Return sample autocorrelations at all lags from 0 to the length
    of the sequence minus 1 for the specified sequence.  The returned
    vector will thus be the same size as the input vector.

    Algorithmically, this function calls NumPy's fast Fourier transform
    and inverse fast Fourier transforms.

    Parameters:
        chain: sequence whose autocorrelation is returned

    Returns:
        autocorrelation estimates at all lags for the specified sequence

    Raises:
        ValueError: if the size of the chain is less than 2
    """
    if len(chain) < 2:
        raise ValueError(f"autocorr requires len(chain) >= 2, but {len(chain)=}")
    chain = np.asarray(chain)
    size = 2 ** np.ceil(np.log2(2 * len(chain) - 1)).astype("int")
    var = np.var(chain)
    ndata = chain - np.mean(chain)
    fft = np.fft.fft(ndata, size)
    sq_mag = np.abs(fft) ** 2
    N = len(ndata)
    acorr = np.fft.ifft(sq_mag).real / var / N
    return acorr[0:N]
