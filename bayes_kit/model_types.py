from typing import Protocol, Tuple
from numpy.typing import ArrayLike, NDArray
import numpy as np


class LogDensityModel(Protocol):
    def dims(self) -> int:
        """number of parameters"""
        ...

    def log_density(self, params_unc: ArrayLike) -> float:
        """unnormalized log density"""
        ...


class GradModel(LogDensityModel, Protocol):
    def log_density_gradient(
        self, params_unc: ArrayLike
    ) -> Tuple[float, NDArray[np.float64]]:
        ...


class HessianModel(GradModel, Protocol):
    def log_density_hessian(
        self, params_unc: ArrayLike
    ) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        ...
