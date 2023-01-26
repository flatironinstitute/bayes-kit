from typing import Union
from scipy import stats
import numpy as np
import numpy.typing as npt


class BetaBinom:
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

    def log_density(self, theta: npt.NDArray[np.float64]) -> float:
        return self.log_likelihood(theta) + self.log_prior(theta)

    def log_prior(self, theta: npt.NDArray[np.float64]) -> float:
        return stats.beta.logpdf(theta[0], self.alpha, self.beta)

    def log_likelihood(self, theta: npt.NDArray[np.float64]) -> float:
        return stats.binom.logpmf(self.x, self.N, theta[0])

    def initial_state(self, _: int) -> npt.NDArray[np.float64]:
        return self._rand.beta(self.alpha, self.beta, size=1)

    def log_density_gradient(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        # use finite diffs for now
        epsilon = 0.000001

        lp = self.log_density(theta)
        lp_plus_e = self.log_density(theta + epsilon)
        return lp, np.array([(lp - lp_plus_e)])

    def posterior_mean(self) -> float:
        return (self.alpha + self.x) / (self.alpha + self.beta + self.N)

    def posterior_variance(self) -> float:
        return ((self.alpha + self.x) * (self.beta + self.N - self.x)) / (
            (self.alpha + self.beta + self.N) ** 2
            * (self.alpha + self.beta + self.N + 1)
        )
