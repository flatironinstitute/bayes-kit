from typing import Optional

import numpy as np
from scipy import stats as sst
from scipy.special import expit as inv_logit
from scipy.special import log1p, logit

from bayes_kit.typing import Seed, VectorType


class Binomial:
    """
    Binomial model with a conjugate beta prior.

    The posterior has a closed form beta distribution.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        x: int,
        N: int,
        seed: Optional[Seed] = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.x = x
        self.N = N
        self._rng = np.random.default_rng(seed)
        self._posterior = sst.beta(alpha + x, beta + N - x)

    def dims(self) -> int:
        return 1

    def log_density(self, params_unc: VectorType) -> float:
        """
        This model's joint density is factored as the product of
        the prior density and the likelihood for testing with
        samplers that require that (e.g., SMC).

        On the log scale, this means adding the log_prior and log_likelihood terms.
        """
        return self.log_likelihood(params_unc) + self.log_prior(params_unc)

    def log_prior(self, params_unc: VectorType) -> float:
        theta: float = inv_logit(params_unc[0])
        jac_adjust: float = np.log(theta) + log1p(-theta)
        prior: float = sst.beta.logpdf(theta, self.alpha, self.beta)
        return prior + jac_adjust

    def log_likelihood(self, params_unc: VectorType) -> float:
        theta = inv_logit(params_unc[0])
        return sst.binom.logpmf(self.x, self.N, theta)  # type: ignore # scipy is not typed

    def initial_state(self, _: int) -> VectorType:
        return logit(self._rng.beta(self.alpha, self.beta, size=1))  # type: ignore # scipy is not typed

    def log_density_gradient(self, params_unc: VectorType) -> tuple[float, VectorType]:
        # use finite diffs for now
        epsilon = 0.000001

        lp = self.log_density(params_unc)
        lp_plus_e = self.log_density(params_unc + epsilon)
        return lp, np.array([(lp - lp_plus_e) / epsilon])

    def constrain_draws(self, draws: VectorType) -> VectorType:
        return inv_logit(draws)  # type: ignore # scipy is not typed

    def posterior_mean(self) -> float:
        return self._posterior.mean()  # type: ignore # scipy is not typed

    def posterior_variance(self) -> float:
        return self._posterior.var()  # type: ignore # scipy is not typed
