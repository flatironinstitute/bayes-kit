from typing import Tuple
from scipy import stats
import numpy as np


class BetaBinom:
    def dims(self) -> int:
        return 1

    def log_density(self, theta) -> float:
        return self.log_likelihood(theta) + self.log_prior(theta)

    def log_prior(self, theta) -> float:
        return stats.beta.logpdf(theta[0], 2, 3)

    def log_likelihood(self, theta) -> float:
        return stats.binom.logpmf(5, 15, theta[0])

    def initial_state(self, _: int):
        return stats.beta.rvs(2, 3, size=1)

    def log_density_gradient(self, theta) -> Tuple[float, np.ndarray]:
        # use finite diffs for now
        epsilon = 0.000001

        lp = self.log_density(theta)
        lp_plus_e = self.log_density(theta + epsilon)
        return lp, np.array([(lp - lp_plus_e)])
