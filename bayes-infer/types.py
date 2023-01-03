from typing import Protocol, Tuple
from numpy.typing import NDArray
import numpy as np

IntType = int
FloatType = np.float64
VectorType = NDArray[FloatType]
MatrixType = NDArray[FloatType]

# MODEL TYPES

class ParametricModel(Protocol):
    def dims(self) -> IntType:
        ...

class LogDensityModel(ParametricModel, Protocol):
    def log_density(self, params_unc: VectorType) -> FloatType:
        ...

class GradientModel(LogDensityModel, Protocol):
    def log_density_gradient(
        self, params_unc: VectorType
    ) -> Tuple[FloatType, VectorType]:
        ...

class HessianModel(GradientModel, Protocol):
    def log_density_hessian(
        self, params_unc: VectorType
    ) -> Tuple[FloatType, VectorType, MatrixType]:
        ...


# SAMPLERS AND TUNING PARAMETERS        

class Sampler(Protocol):
    def sample(self) -> VectorType:
        ...
    def __iter__(self) -> Iterator[Vectortype]:
        ...

class Metric(Protocol):
    def transform(self, VectorType) -> VectorType:
        ...

class StepSize(Protocol):
    def step_size(self) -> FloatType:
        ...

class NumSteps(Protocol):
    def num_steps(self) -> IntType:
        ...
