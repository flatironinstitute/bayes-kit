from typing import Union
from scipy import stats
from scipy.special import logit, expit as inv_logit, log1p
import numpy as np
import numpy.typing as npt


class Binomial:
    """
    Binomial model with a (conjugate) beta prior.

    This model is decomposed as a prior and a likelihood for
    testing with samplers that require that, e.g. SMC. The
    posterior has a closed-form beta distribution.
    """

    def __init__(
        self,
        alpha: float = 2,
        beta: float = 3,
        x: int = 5,
        N: int = 15,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.x = x
        self.N = N
        self._rand = np.random.default_rng(seed)

    def dims(self) -> int:
        return 1

    def log_density(self, params_unc: npt.NDArray[np.float64]) -> float:
        return self.log_likelihood(params_unc) + self.log_prior(params_unc)

    def log_prior(self, theta: npt.NDArray[np.float64]) -> float:
        theta_constrained = inv_logit(theta[0])
        jac_adjust: float = np.log(theta_constrained) + log1p(-theta_constrained)
        prior: float = stats.beta.logpdf(theta_constrained, self.alpha, self.beta)
        return prior + jac_adjust

    def log_likelihood(self, theta: npt.NDArray[np.float64]) -> float:
        theta_constrained = inv_logit(theta[0])
        return stats.binom.logpmf(self.x, self.N, theta_constrained)  # type: ignore # scipy is not typed

    def initial_state(self, _: int) -> npt.NDArray[np.float64]:
        return logit(self._rand.beta(self.alpha, self.beta, size=1))  # type: ignore # scipy is not typed

    def log_density_gradient(
        self, params_unc: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        # use finite diffs for now
        epsilon = 0.000001

        lp = self.log_density(params_unc)
        lp_plus_e = self.log_density(params_unc + epsilon)
        return lp, np.array([(lp - lp_plus_e) / epsilon])

    def constrain_draws(
        self, draws: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return inv_logit(draws)  # type: ignore # scipy is not typed

    def posterior_mean(self) -> float:
        return (self.alpha + self.x) / (self.alpha + self.beta + self.N)

    def posterior_variance(self) -> float:
        return ((self.alpha + self.x) * (self.beta + self.N - self.x)) / (
            (self.alpha + self.beta + self.N) ** 2
            * (self.alpha + self.beta + self.N + 1)
        )
