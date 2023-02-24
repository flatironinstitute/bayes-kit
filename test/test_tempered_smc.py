from test.models.binomial import Binomial
from bayes_kit.smc import TemperedLikelihoodSMC, metropolis_kernel
import numpy as np


def test_rwm_smc_binom() -> None:
    model = Binomial(alpha=2, beta=3, x=5, N=15)
    M = 75
    N = 15
    rwm_smc = TemperedLikelihoodSMC(
        M,
        N,
        model.initial_state,
        model.log_likelihood,
        model.log_prior,
        metropolis_kernel(0.5),
    )

    rwm_smc.run()
    draws = model.constrain_draws(rwm_smc.thetas)

    mean = draws.mean(axis=0)
    var = draws.var(axis=0, ddof=1)

    print(f"{draws[1:10]=}")
    print(f"{mean=}  {var=}")

    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.05)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.01)
