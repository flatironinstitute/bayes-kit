from typing import Callable, Iterator

import numpy as np
from numpy.typing import ArrayLike

from .typing import LogPriorLikelihoodModel, VectorType

DensityFunction = Callable[[VectorType], float]
Kernel = Callable[[VectorType, DensityFunction], ArrayLike]


class TemperedLikelihoodSMC:
    def __init__(
        self,
        model: LogPriorLikelihoodModel,
        M: int,
        N: int,
        sample_initial: Callable[[int], ArrayLike],
        kernel: Kernel,
    ) -> None:
        self.M = M
        self.N = N
        self.thetas = np.array(list(map(sample_initial, range(M))))
        self.D = self.thetas.shape[1]

        self._model = model
        self.kernel = kernel

    def log_prior(self, theta: VectorType) -> float:
        return self._model.log_prior(theta)

    def log_likelihood(self, theta: VectorType) -> float:
        return self._model.log_likelihood(theta)

    def __iter__(self) -> Iterator[VectorType]:
        self.run()
        return iter(self.thetas)

    def run(self) -> None:
        for n in range(1, self.N + 1):
            self.transition(n)

    def time(self, n: int) -> float:
        return n / self.N

    def transition(self, n: int) -> None:
        def lpminus1(theta: VectorType) -> float:
            return self.log_likelihood(theta) * self.time(n - 1) + self.log_prior(theta)

        def lp(theta: VectorType) -> float:
            return self.log_likelihood(theta) * self.time(n) + self.log_prior(theta)

        # note: try to do this in parallel
        for m in range(self.M):
            self.thetas[m] = np.asanyarray(
                self.kernel(np.atleast_1d(self.thetas[m]), lpminus1), dtype=np.ndarray
            )

        # note: duplicating calls to lpminus1
        self.thetas = importance_resample(self.thetas, lpminus1, lp)


# TODO(bward): factor out
def importance_resample(
    thetas: VectorType, lpminus1: DensityFunction, lp: DensityFunction
) -> VectorType:
    weights = np.exp(
        np.apply_along_axis(lp, axis=1, arr=thetas)
        - np.apply_along_axis(lpminus1, axis=1, arr=thetas)
    )
    M = thetas.shape[0]
    # TODO(bward): should use random Generator object
    idxs = np.random.choice(M, size=M, replace=True, p=weights / weights.sum())

    return thetas[idxs]  # type: ignore


# TODO(bward): factor out/reuse Metropolis sampler
def metropolis_kernel(scale: float) -> Kernel:
    def proposal_rng(theta: VectorType) -> VectorType:
        return np.random.normal(loc=theta, scale=scale)

    def metropolis(theta: VectorType, lp: DensityFunction) -> VectorType:
        theta_star = proposal_rng(theta)
        if np.log(np.random.uniform()) < lp(theta_star) - lp(theta):
            return theta_star
        return theta

    return metropolis
