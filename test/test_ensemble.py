from test.models.std_normal import StdNormal
from bayes_kit.ensemble import AffineInvariantWalker
import numpy as np

def test_aiw_std_normal() -> None:
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    sampler = AffineInvariantWalker(model, a = 2, walkers=10)
    M = 10
    for m in range(M):
        theta = sampler.sample()
        print(theta)
    return 1
    draws = np.array([sampler.sample()[0] for _ in range(M)])
    print(f"{draws=}")
    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)
    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.1)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.1)
