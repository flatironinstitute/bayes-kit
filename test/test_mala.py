from models.std_normal import StdNormal
from bayes_infer.mala import MALA
import numpy as np

def test_mala():
    # init with draw from posterior
    init = np.random.normal(loc=0, scale=1, size=[1])
    model = StdNormal()
    mala = MALA(model, 0.3, init)

    M = 1000
    draws = np.array([mala.sample()[0] for _ in range(M)])

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    np.testing.assert_allclose(mean, 0, atol=0.1)
    np.testing.assert_allclose(var, 1, atol=0.1)

    print(f"{draws[1:10]=}")
    print(f"{mean=}  {var=}")


