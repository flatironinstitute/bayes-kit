from test.models.std_normal import StdNormal
from bayes_kit.ensemble import AffineInvariantWalker
import numpy as np

def test_aiw_std_normal() -> None:
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    sampler = AffineInvariantWalker(model, a = 2, num_walkers=8)
    D = sampler._dim
    K = sampler._num_walkers
    M = 1000
    draws = np.ndarray(shape=(M, K, D))
    for m in range(M):
        draws[m, 0:K, 0:D] = sampler.sample()
    mean = np.mean(draws)
    var = np.var(draws, ddof=1)
    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)
