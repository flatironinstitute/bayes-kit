from typing import Callable, Optional, Tuple
from numpy.typing import NDArray
import numpy as np

from .model_types import LogDensityModel


class AffineInvariantWalker:
    """
    An implementation of the affine-invariant ensemble sampler of
    Goodman and Weare (2010).  

    References:
    Goodman, J. and Weare, J., 2010. Ensemble samplers with affine invariance.
    *Communications in Applied Mathematics and Computational Science*
    5(1):65--80.
    """

    def __init__(
            self,
            model: LogDensityModel,
            a: Optional[float] = None,
            walkers: Optional[int] = None,
            init: Optional[NDArray[np.float64]] = None
        ):
        """
        Initialize the sampler with a log density model, and optionally
        proposal bounds, number of walkers and initial parameter values.
            
        Parameters:
        model: class used to evaluate log densities
        a: bounds on proposal (default 1)
        walkers: an even number of walkers to use (default dimensionality of `model * 2`)
        init: `walker` x `dimensio`n array of initial positions (defaults to standard normal)

        Throws:
        ValueError: if `a` is provided and not >= 1, `walker`s is provided and not strictly positive and even,
        or if the `init` is provided and is not an `NDArray` of shape `walker` x `dimension`
        """
        self._model = model
        self._dim = self._model.dims()
        if a != None and a < 1:
            raise ValueError(f"stretch bound must be greater than or equal to 1; found {a=}")
        self._a = a or 1
        self._sqrt_a = np.sqrt(a)
        self._inv_sqrt_a = 1 / self._sqrt_a
        if walkers != None and (walkers < 2 or walkers % 2 != 0) :
            raise ValueError(f"walkers must be strictly positive, even integer; found {walkers=}")
        self._walkers = walkers or 2 * self._dim
        self._halfwalkers = self._walkers // 2
        self._drawshape = (self._walkers, self._dim)
        if init != None and init.shape != self._drawshape:
            raise ValueError(f"init must be shape of draw {self._drawshape}; found {init.shape=}")
        self._thetas = init or np.random.normal(size=self._drawshape)
        self._firsthalf = range(0, self._halfwalkers)
        self._secondhalf = range(self._halfwalkers, self._walkers)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample

    def draw_z(self):
        """Return random draw z in (1/a, a) with p(z) propto 1 / sqrt(z)"""
        return np.square(np.random.uniform(self._inv_sqrt_a, self._sqrt_a))

    def stretch_move(self, theta_k: NDArray[np.float64], theta_j: NDArray[np.float64]):
        z = self.draw_z()
        theta_star = theta_j + z * (theta_k - theta_j)  # (1 - z) * theta_j + z * theta_k
        print(f"{theta_k=}  {theta_j=}  {z=}  {theta_star=}")
        log_q = (self._dim - 1) * np.log(z) + self._model.log_density(theta_star) - self._model.log_density(theta_k)
        log_u = np.log(np.random.uniform())
        print(f"{log_q=}  {log_u=}")
        if log_u < log_q:
            return theta_star
        return theta_k

    def sample(self) -> NDArray[np.float64]:
        print(f"IN: {self._thetas=}")
        js = np.random.choice(self._secondhalf, size=self._halfwalkers, replace=False)
        for k in self._firsthalf:
            self._thetas[k] = self.stretch_move(self._thetas[k], self._thetas[js[k]])
        js = np.random.choice(self._firsthalf, size=self._halfwalkers, replace=False)
        for k in self._secondhalf:
            self._thetas[k] = self.stretch_move(self._thetas[k], self._thetas[js[k - self._halfwalkers]])
        print(f"OUT: {self._thetas=}")
        return self._thetas


