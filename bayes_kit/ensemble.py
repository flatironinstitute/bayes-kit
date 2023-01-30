from typing import Any, Callable, Iterator, Optional, Tuple
from numpy.typing import NDArray
import numpy as np

from .model_types import LogDensityModel

Sample = NDArray[np.float64]

class AffineInvariantWalker:
    """The affine-invariant ensemble of Goodman and Weare (2010).  

    References:
        Goodman, J. and Weare, J., 2010. Ensemble samplers with affine invariance.
        *Communications in Applied Mathematics and Computational Science*
        5(1):65--80.

    Attributes:
        _model (LogDensityModel): The statistical model being sampled.
        _dim (int): The number of model dimensions.
        _a (np.float64): The upper bound of interpolation ratio sampling (lower bound is inverse).
        _sqrt_a (np.float64): The square root of `_a`.
        _inv_sqrt_a (np.float64): The inverse square root of `_a`.
        _walkers (np.int64): The number of ensemble members.
        _half_walkers (np.int64): Half the number of walkers.
        _drawshape (list(int)): The number of walks by number of dimensions.
        _thetas (NDArray[np.float64]): The ensemble of draws (`_walkers` x `_dim`).
        _lps (NDArray[np.float64]): The vector of log densities (`_walkers x 1`).
        _firsthalf (NDArray[np.float64]): A view of the first half of `_thetas`.
        _secondhalf (NDArray[np.float64]): A view of the second half of `_thetas`.
    """
    def __init__(
            self,
            model: LogDensityModel,
            a: Optional[float] = None,
            walkers: Optional[int] = None,
            init: Optional[NDArray[np.float64]] = None
        ):
        """Initialize the sampler with model, and optionally bounds, size, and initial values.
        
        The class instance stores the model, bounds on the proposal on the square root scale,
        and the walkers.  The initialization is used for the value of the parameters *before* the
        first draw.  The initialization will *not* be returned as one of the draws.

        Args:
            model (LogDensityModel): class used to evaluate log densities
            a (float): bounds on proposal (default 2)
            walkers (int): an even number of walkers to use (default dimensionality of `model * 2`)
            init (NDArray[np.float64]): `walker` x `dimension` array of initial positions

        Raises:
        ValueError: If `a` is provided and is not greater than or equal to 1, `walker`s is provided and not strictly positive and even, or if the `init` is provided and is not an `NDArray` of shape `walker` x `dimension`
        """
        self._model = model
        self._dim = self._model.dims()
        if a != None and np.float64(a) < 1:
            raise ValueError(f"stretch bound must be greater than or equal to 1; found {a=}")
        self._a = np.float64(a or 2.0)
        self._sqrt_a = np.sqrt(np.float64(a))
        self._inv_sqrt_a = 1 / self._sqrt_a
        self._walkers = np.int64(walkers or 2 * self._dim)
        if self._walkers < 2 or self._walkers % 2 != 0:
            raise ValueError(f"walkers must be strictly positive, even integer; found {walkers=}")
        self._halfwalkers = self._walkers // 2
        self._drawshape = (int(self._walkers), self._dim)
        self._thetas = np.asarray(init or np.random.normal(size=self._drawshape))
        self._lps = [self._model.log_density(theta) for theta in self._thetas]
        if self._thetas.shape != self._drawshape:
            raise ValueError(f"init must be shape of draw {self._drawshape}; found {self._thetas.shape=}")
        self._firsthalf = range(0, int(self._halfwalkers))
        self._secondhalf = range(int(self._halfwalkers), int(self._walkers))

    def __iter__(self) -> Iterator[Sample]:
        """Return an infinite iterator for sampling.

        Returns:
            An iterator generating samples.
        """
        return self

    def __next__(self) -> Sample:
        """Return the next sample.

        Returns:
            The next sample.
        """
        return self.sample()

    def draw_z(self) -> Sample:
        """
        Return a random draw of `z` in `(1/a, a)` with `p(z) propto 1 / sqrt(z)`.

        Returns:
            A random draw of `z`.
        """
        draw: NDArray[np.float64] = np.square(np.random.uniform(self._inv_sqrt_a, self._sqrt_a))
        return draw

    def stretch_move(self, k: int, j: int) -> Any:
        theta_k = self._thetas[k]
        lp_theta_k = self._lps[k]
        theta_j = self._thetas[j]
        z = self.draw_z()
        theta_star = np.asarray(theta_j + z * (theta_k - theta_j))  # (1 - z) * theta_j + z * theta_k
        lp_theta_star = self._model.log_density(theta_star)
        log_q = (self._dim - 1) * np.log(z) + lp_theta_star - lp_theta_k
        log_u = np.log(np.random.uniform())
        if log_u < log_q:
            self._thetas[k] = theta_star
            self._lps[k] = lp_theta_star

    def sample(self) -> Sample:
        js = np.random.choice(self._secondhalf, size=self._halfwalkers, replace=False)
        for k in self._firsthalf:
            self.stretch_move(k, js[k])
        js = np.random.choice(self._firsthalf, size=self._halfwalkers, replace=False)
        for k in self._secondhalf:
            self.stretch_move(k, js[k - self._halfwalkers])
        return self._thetas


