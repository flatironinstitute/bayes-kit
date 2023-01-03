from types import LogDensityModel, VectorType, FloatType
from typing import Tuple
import numpy as np

class StandardNormalModel(GradientModel):
    def log_density(self, theta: VectorType) -> FloatType:
        return -0.5 * np.dot(theta, theta)

    def log_density_gradient(
            self, theta: VectorType
            ) -> Tuple[FloatType, VectorType]:
        return log_density(self, theta), -theta
