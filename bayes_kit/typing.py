from typing import Protocol, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Generic types used in most modules
FloatType = np.float64
IntType = np.int64
VectorType = NDArray[FloatType]
DrawAndLogP = tuple[VectorType, float]
Seed = Union[int, np.random.BitGenerator, np.random.Generator]
ChainType = Union[Sequence[float], VectorType]


class LogDensityModel(Protocol):
    def dims(self) -> int:
        """number of parameters"""
        ...  # pragma: no cover

    def log_density(self, params_unc: VectorType) -> float:
        """unnormalized log density"""
        ...  # pragma: no cover


class GradModel(LogDensityModel, Protocol):
    def log_density_gradient(self, params_unc: VectorType) -> Tuple[float, ArrayLike]:
        ...  # pragma: no cover


class HessianModel(GradModel, Protocol):
    def log_density_hessian(
        self, params_unc: VectorType
    ) -> Tuple[float, ArrayLike, ArrayLike]:
        ...  # pragma: no cover


class LogPriorLikelihoodModel(LogDensityModel, Protocol):
    def log_prior(self, params_unc: VectorType) -> float:
        ...  # pragma: no cover

    def log_likelihood(self, params_unc: VectorType) -> float:
        ...  # pragma: no cover
