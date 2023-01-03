from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray

DensityFunction = Callable[[NDArray[np.float64]], float]


class TemperedLikelihoodSMC:
    def __init__(
        self,
        M: int,
        N: int,
        sample_initial: Callable[[int], ArrayLike],
        log_likelihood: DensityFunction,
        log_prior: DensityFunction,
        kernel: Callable[[NDArray[np.float64], DensityFunction], ArrayLike],
    ) -> None:

        self.M = M
        self.N = N
        self.thetas = np.array([sample_initial(m) for m in range(M)])
        self.D = len(self.thetas[0])

        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.kernel = kernel

    def __iter__(self):
        self.run()
        return iter(self.thetas)

    def run(self):
        for n in range(1, self.N + 1):
            self.transition(n)

    def transition(self, n: int) -> None:
        tminus1 = (n - 1) / self.N

        def lpminus1(theta):
            return self.log_likelihood(theta) * tminus1 + self.log_prior(theta)

        t = n / self.N

        def lp(theta):
            return self.log_likelihood(theta) * t + self.log_prior(theta)

        # note: try to do this in parallel
        for m in range(self.M):
            self.thetas[m] = np.asanyarray(
                self.kernel(np.atleast_1d(self.thetas[m]), lpminus1), dtype=np.ndarray
            )
        # note: duplicating calls to lpminus1
        self.thetas = importance_resample(self.thetas, lpminus1, lp)


def importance_resample(
    thetas: NDArray[np.float64], lpminus1: DensityFunction, lp: DensityFunction
):
    weights = np.exp(
        np.apply_along_axis(
            lambda theta: lp(theta) - lpminus1(theta), axis=1, arr=thetas
        )
    )
    # note: should use Generator object
    M = thetas.shape[0]
    idxs = np.random.choice(M, size=M, replace=True, p=weights / weights.sum())
    return thetas[idxs]


def metropolis_kernel(scale: float):
    def proposal_rng(theta: NDArray[np.float64]):
        return np.random.normal(loc=theta, scale=scale)

    def metropolis(theta: NDArray[np.float64], lp: DensityFunction):
        theta_star = proposal_rng(theta)
        if np.log(np.random.uniform()) < lp(theta_star) - lp(theta):
            return theta_star
        return theta

    return metropolis
