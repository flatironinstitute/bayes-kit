from models.std_normal import StdNormal
from bayes_infer.hmc import HMCDiag
import numpy as np

def test_hmc_diag():
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    mala = HMCDiag(model, steps=10, stepsize=0.25, init=init)

    M = 10000
    draws = np.array([mala.sample()[0] for _ in range(M)])

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, 0, atol=0.1)
    np.testing.assert_allclose(var, 1, atol=0.1)
