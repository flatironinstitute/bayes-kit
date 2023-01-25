from typing import Tuple
from scipy import stats
import numpy as np


X = 5
N = 15
A = 2
B = 3


class BetaBinom:
    def dims(self) -> int:
        return 1

    def log_density(self, theta) -> float:
        return self.log_likelihood(theta) + self.log_prior(theta)

    def log_prior(self, theta) -> float:
        return stats.beta.logpdf(theta[0], A, B)

    def log_likelihood(self, theta) -> float:
        return stats.binom.logpmf(X, N, theta[0])

    def initial_state(self, _: int):
        return stats.beta.rvs(A, B, size=1)

    def log_density_gradient(self, theta) -> Tuple[float, np.ndarray]:
        # use finite diffs for now
        epsilon = 0.000001

        lp = self.log_density(theta)
        lp_plus_e = self.log_density(theta + epsilon)
        return lp, np.array([(lp - lp_plus_e)])

    def posterior_mean(self):
        return (A + X) / (A + B + N)

    def posterior_variance(self):
        return ((A + X) * (B + N - X)) / ((A + B + N) ** 2 * (A + B + N + 1))
