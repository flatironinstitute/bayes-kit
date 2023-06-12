from typing import Protocol, Tuple
from .numeric import ArrayType, ArrayLike

class LogDensityModel(Protocol):
    def dims(self) -> int:
        """number of parameters"""
        ...  # pragma: no cover

    def log_density(self, params_unc: ArrayType) -> float:
        """unnormalized log density"""
        ...  # pragma: no cover


class GradModel(LogDensityModel, Protocol):
    def log_density_gradient(
        self, params_unc: ArrayType
    ) -> Tuple[float, ArrayLike]:
        ...  # pragma: no cover


class HessianModel(GradModel, Protocol):
    def log_density_hessian(
        self, params_unc: ArrayType
    ) -> Tuple[float, ArrayLike, ArrayLike]:
        ...  # pragma: no cover


class LogPriorLikelihoodModel(LogDensityModel, Protocol):
    def log_prior(self, params_unc: ArrayType) -> float:
        ...  # pragma: no cover

    def log_likelihood(self, params_unc: ArrayType) -> float:
        ...  # pragma: no cover


__all__ = [
    "LogDensityModel",
    "GradModel",
    "HessianModel",
    "LogPriorLikelihoodModel",
]
