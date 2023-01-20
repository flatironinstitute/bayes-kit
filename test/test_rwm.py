from models.std_normal import StdNormal
from bayes_kit.rwm import RandomWalkMetropolis
import numpy as np

def test_rwm():
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    proposal_rng = lambda theta: np.random.normal(loc=theta, scale=4)
    rwm = RandomWalkMetropolis(model, proposal_rng, init)
    M = 10000
    draws = np.array([rwm.sample()[0] for _ in range(M)])
    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)
    np.testing.assert_allclose(mean, 0, atol=0.1)
    np.testing.assert_allclose(var, 1, atol=0.1)

    accept = M - (draws[:M-1] == draws[1:]).sum()
    print(f"{accept=}")
    print(f"{draws[1:10]=}")
    print(f"{mean=}  {var=}")


