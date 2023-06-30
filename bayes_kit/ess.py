import numpy as np
import numpy.typing as npt

import bayes_kit.autocorr as autocorr
from bayes_kit.iat import iat, iat_imse, iat_ipse

FloatType = np.float64
IntType = np.int64
VectorType = npt.NDArray[FloatType]


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
        raise ValueError(f"ess_ipse(chain) requires len(chain) >= 4, but {len(chain)=}")
    return len(chain) / iat_ipse(chain)


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
        raise ValueError(
            f"ess_imse(chain) requires len(chain) >=4, but {len(chain) = }"
        )
    return len(chain) / iat_imse(chain)


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
    if len(chain) < 4:
        raise ValueError(f"ess(chain) requires len(chain) >=4, but {len(chain) = }")
    return len(chain) / iat(chain)
