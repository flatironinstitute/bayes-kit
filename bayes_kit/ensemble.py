from typing import Any, Iterator, Optional, Union
from numpy.typing import NDArray
import numpy as np

from .model_types import LogDensityModel

Sample = NDArray[np.float64]

class AffineInvariantWalker:
    """The affine-invariant ensemble sampler with stretch updates.

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
        _num_walkers (np.int64): The number of ensemble members.
        _half_num_walkers (np.int64): Half the number of walkers.
        _drawshape (list(int)): The number of walks by number of dimensions.
        _thetas (NDArray[np.float64]): The ensemble of draws (`_num_walkers` x `_dim`).
        _lp_thetas (NDArray[np.float64]): The vector of log densities (`_num_walkers x 1`).
        _first_range (NDArray[np.float64]): Range of indexes of first half of `_thetas`.
        _second_range (NDArray[np.float64]): Range of indexes for second half of `_thetas`.
        _rng (np.random.Generator): pseudo random number generator

    """
    
    def __init__(
            self,
            model: LogDensityModel,
            a: Optional[float] = None,
            num_walkers: Optional[int] = None,
            init: Optional[NDArray[np.float64]] = None,
            seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
        ):
        """Initialize the sampler with model, and optionally bounds, size, and initial values.
        
        The class instance stores the model, bounds on the proposal on
        the square root scale, and the number of walkers.  The
        initialization is used for the value of the parameters *before*
        the first draw; the initialization will *not* be returned as
        one of the draws.

        Arguments:
            model (LogDensityModel): class used to evaluate log densities
            a (float): The bounds on the interpolation ratio proposal (default 2)
            walkers (int): An even number of walkers to use (default dimensionality of `model * 2`)
            init (NDArray[np.float64]): `walker` x `dimension` array of initial positions.
            seed (Union[None, int, np.random.BitGenerator, np.random.Generator]): Pseudo-RNG seed or generator.

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
        self._num_walkers = num_walkers or 2 * self._dim
        if self._num_walkers < 2 or self._num_walkers % 2 != 0:
            raise ValueError(f"number of walkers must be strictly positive, even integer; found {num_walkers=}")
        self._half_num_walkers = self._num_walkers // 2
        self._drawshape = (int(self._num_walkers), self._dim)
        self._rng = np.random.default_rng(seed)
        self._thetas : NDArray[np.float64] = init or self._rng.normal(size=self._drawshape)
        self._lp_thetas = [self._model.log_density(theta) for theta in self._thetas]
        if self._thetas.shape != self._drawshape:
            raise ValueError(f"init must be shape of draw {self._drawshape}; found {self._thetas.shape=}")
        self._first_range = range(0, int(self._half_num_walkers))
        self._second_range = range(int(self._half_num_walkers), int(self._num_walkers))

    def __iter__(self) -> Iterator[Sample]:
        """Return an infinite iterator for ensemble sampling.

        Returns:
            An iterator generating samples.
        """
        return self

    def __next__(self) -> Sample:
        """Return the next ensemble sample (`_num_walkers` x `_dim`).

        Returns:
            The next sample.
        """
        return self.sample()

    def draw_z(self) -> Sample:
        """Return a random draw of `z` in `(1/a, a)` with `p(z) propto 1 / sqrt(z)`.

        Returns:
            A random draw of `z`.
        """
        draw: NDArray[np.float64] = np.square(self._rng.uniform(self._inv_sqrt_a, self._sqrt_a))
        return draw

    def stretch_move(self, k: int, j: int) -> Any:
        """Update the walkers with a single stretch move.

        Arguments:
            k (int): walker to update
            j (int): complementary walker with which to interpolate/extrapolate
        """
        theta_k = self._thetas[k]
        lp_theta_k = self._lp_thetas[k]
        theta_j = self._thetas[j]
        z = self.draw_z()
        theta_star: NDArray[np.float64] = theta_j + z * (theta_k - theta_j)
        lp_theta_star = self._model.log_density(theta_star)
        log_q = (self._dim - 1) * np.log(z) + lp_theta_star - lp_theta_k
        log_u = np.log(self._rng.uniform())
        if log_u < log_q:
            self._thetas[k] = theta_star
            self._lp_thetas[k] = lp_theta_star

    def sample(self) -> Sample:
        """Return an ensemble draw (`_num_walkers` x `_dim`).

        Returns:
            An ensemble draw.
        """
        js = self._rng.choice(self._second_range, size=self._half_num_walkers, replace=False)
        for k in self._first_range:
            self.stretch_move(k, js[k])
        js = self._rng.choice(self._first_range, size=self._half_num_walkers, replace=False)
        for k in self._second_range:
            self.stretch_move(k, js[k - self._half_num_walkers])
        return self._thetas


