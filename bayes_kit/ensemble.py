import numpy as np

from bayes_kit.types import LogDensityModel, ArrayType
from typing import Optional

class Stretcher:
    """
    Goodman, J. and Weare, J., 2010. Ensemble samplers with affine invariance.
    *Communications in Applied Mathematics and Computational Science*
    5(1):65--80.
    """

    # def __init__(
    #         self,
    #         model: LogDensityModel,
    #         a: Optional[float] = None,
    #         walkers: Optional[int] = None
    #         init: Optional[ArrayType] = None)
    #     ):
    #     self._model = model
    #     self._dim = self._model.dims()
    #     if a != None and a < 1:
    #         raise ValueError(f"stretch bound must be greater than or equal to 1; found {a=}")
    #     self._a = a
    #     self._sqrt_a = np.sqrt(a)
    #     self._inv_sqrt_a = 1 / self._sqrt_a
    #     if walkers != NONE and (walkers <= 0 or walkers % 2 != 0) :
    #         raise ValueError(f"walkers must be strictly positive, even integer; found {walkers=}")
    #     self._walkers = walkers or 2 * self._dim
    #     self._halfwalkers = a / 2
    #     self._drawshape = (self._walkers, self._dim)
    #     if init != None and init.shape != self._drawshape:
    #         raise ValueError(f"init must be shape of draw {self._drawshape}; found {init.shape=}")
    #     self._thetas = init if init is not None else np.random.normal(size=self._drawshape)
    #     self._firsthalf = range(halfwalkers)
    #     self._secondhalf = range(halfwalkers, walkers)

    # def __iter__(self):
    #     return self

    # def __next__(self):
    #     return self.sample

    # def draw_z(self):
    #     """Return random draw z in (1/a, a) with p(z) propto 1 / sqrt(z)"""
    #     return np.square(np.random.uniform(self._inv_sqrt_a, self._sqrt_a))

    # def stretch_move(self, theta_k: ArrayType, theta_j: ArrayType):
    #     z = self.draw_z()
    #     theta_star = theta_j + z * (theta_k - theta_j)  # (1 - z) * theta_j + z * theta_k
    #     log_q = (self._dims - 1) * np.log(z) + self._model.log_density(theta_star) - self._model.log_density(theta_k)
    #     if np.log(np.random.uniform()) < log_q:
    #         return theta_star
    #     return theta_k

    # def sample(self) -> ArrayType
    #     js = np.random.choice(secondhalf, size=self._halfwalkers)
    #     for k in firsthalf:
    #         self._thetas[k] = stretch_move(self._thetas[k], self._thetas[js[k]])
    #     js = np.random.choice(firsthalf, size=self._halfwalkers)
    #     for k in secondhalf:
    #         self_thetas[k] = stretch_move(self._thetas[k], self._thetas[js[k]])
    #     return self._thetas


# TODO(carpenter): cache log density rather than recomputing for self
