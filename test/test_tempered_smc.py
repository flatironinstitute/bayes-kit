from models.beta_binomial import BetaBinom
from bayes_infer.smc import TemperedLikelihoodSMC, metropolis_kernel
import numpy as np


def test_rwm_smc():
    model = BetaBinom()
    M = 75
    N = 10
    rwm_smc = TemperedLikelihoodSMC(
        M,
        N,
        model.initial_state,
        model.log_likelihood,
        model.log_prior,
        metropolis_kernel(0.3),
    )

    rwm_smc.run()
    draws = rwm_smc.thetas

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    print(f"{draws[1:10]=}")
    print(f"{mean=}  {var=}")

    np.testing.assert_allclose(mean, 7 / 25, atol=0.1)
    np.testing.assert_allclose(var, 7 * 18 / (25**2 * 26), atol=0.1)
