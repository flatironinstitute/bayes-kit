from typing import Iterator, Optional, Union, Callable
from numpy.typing import NDArray
import numpy as np

from numpy.linalg import norm
from .model_types import GradModel

Draw = tuple[NDArray[np.float64], float]


def validate_stepsize(stepsize: float) -> None:
    """Ensure that leapfrog stepsize is positive.

    Args:
        stepsize: stepsize in each leapfrog step

    Raises:
        ValueError: stepsize is not positive
    """
    try:
        assert stepsize > 0
    except:
        raise ValueError("stepsize must be positive")


def validate_steps(steps: int) -> None:
    """Ensure that number of leapfrog steps is positive and an integer.

    Args:
        steps: number of steps to run the leapfrog integrator

    Raises:
        ValueError: steps is not positive
        TypeError: steps is not an integer
    """
    try:
        assert steps > 0
    except:
        raise ValueError("steps must be positive")
    try:
        assert steps == int(steps)
    except:
        raise TypeError("steps must be an integer")


class DrHmcDiag:
    """Delayed Rejection Hamiltonian Monte Carlo sampler with diagonal metric.

    To sample from a target distribution, DrHmcDiag proposes a new sample from the
    current sample using a leapfrog integrator. The probability of accepting this
    transition is recursively computed. If accepted, DrHmcDiag returns the proposed
    sample and its log density. If rejected, DrHmcDiag proposes a new sample generated
    with a smaller step size for the leapfrog integrator. This process is repeated
    until a maximum number of proposals is reached or probabilistic delayed rejection
    determines not to propose a new sample.

    Implementation of DrHmcDiag is based on 'Delayed rejection Hamiltonian Monte Carlo
    for sampling multiscale distirbutions' by Chirag et al. While the paper represents
    position and momentum with (p, q), this implementation uses (theta, rho) for
    consistency with bayes-kit.

    Probabilistic delayed rejection proposes a new sample with some probability
    dependent on where we are in the distribution. If a proposal has low acceptance
    probability and was rejected, propose another sample; if a proposal has high
    acceptance probablity but was rejected by chance, do not propose another
    sample. More details found in section 3.2 of Chirag et al.
    """

    def __init__(
        self,
        model: GradModel,
        stepsize: Union[float, Callable[[int], float]],
        steps: Union[float, Callable[[int], int]],
        num_proposals: int = 2,
        metric_diag: Optional[NDArray[np.float64]] = None,
        init: Optional[NDArray[np.float64]] = None,
        seed: Union[None, int, np.random.BitGenerator, np.random.Generator] = None,
    ):
        """Initialize the DrHmcDiag sampler.

        The stepsize argument can be a float, for a fixed stepsize, or a function, for 
        a stepsize that depends on the proposal number. The stepsize must be positive.
        
        The steps argument can be an integer, for a fixed number of steps, or a 
        function, for a number of steps that depends on the proposal number. The number
        of steps must be positive and an integer.

        Args:
            model: model with log density and gradient
            stepsize: stepsize in each leapfrog step
            steps: number of leapfrog steps
            num_proposals: number of delayed rejection proposals. Defaults to 3.
            metric_diag: diagonal of the metric. Defaults to None.
            init: initial value of the Markov chain. Defaults to None.
            seed: seed for numpy rng. Defaults to None.
        """

        self._model = model
        self._dim = self._model.dims()
        self._stepsize = stepsize if callable(stepsize) else lambda k: stepsize
        self._steps = steps if callable(steps) else lambda k: steps
        self._num_proposals = num_proposals
        self._metric = metric_diag or np.ones(self._dim)
        self._rng = np.random.default_rng(seed)
        self._theta = (
            init
            if (init is not None and init.shape != (0,))
            else self._rng.normal(size=self._dim)
        )
        self._stepsize_list = []
        self._steps_list = []

    def __iter__(self) -> Iterator[Draw]:
        """Use the DrHmcDiag sampler as an iterator.

        Returns:
            Iterator[Draw]: Iterator of draws from DrHmcDiag sampler
        """
        return self

    def __next__(self) -> Draw:
        """Yields next draw from DrHmcDiag iterator.

        Yields:
            Draw: Tuple of (sample, log density)
        """
        return self.sample()

    def joint_logp(self, theta: NDArray[np.float64], rho: NDArray[np.float64]) -> float:
        """Joint log density of sample (theta, rho) under the cannonical distribution.

        Args:
            theta: position
            rho: momentum

        Returns:
            float: joint log density
        """
        adj: float = -0.5 * np.dot(rho, self._metric * rho)
        return self._model.log_density(theta) + adj

    def leapfrog(
        self,
        theta: NDArray[np.float64],
        rho: NDArray[np.float64],
        stepsize: float,
        steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Propose new sample (theta_prop, rho_prop) with leapfrog integrator.

        Discretize and solve Hamilton's equations using the leapfrog integrator to
        deterministically propose a new sample (theta_prop, rho_prop) from the current
        sample (theta, rho).

        Copy theta because numpy's += operator mutates the original array instead of
        creating a new one.

        Initialize rho_mid by going backwards half a step so that the first full-step
        inside the loop brings rho_mid up to +1/2 steps.

        Args:
            theta: position
            rho: momentum
            stepsize: stepsize in each leapfrog step
            steps: number of leapfrog steps

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: proposed sample
        """
        theta = np.array(theta, copy=True)
        _, grad = self._model.log_density_gradient(theta)
        rho_mid = rho - 0.5 * stepsize * np.multiply(self._metric, grad)
        for n in range(steps):
            rho_mid += stepsize * np.multiply(self._metric, grad)
            theta += stepsize * rho_mid
            _, grad = self._model.log_density_gradient(theta)
        rho = rho_mid + 0.5 * stepsize * np.multiply(self._metric, grad)
        return (theta, rho)

    def get_stepsize(self, k) -> float:
        """Computes new leapfrog stepsize for delayed rejection proposal.

        Stores new stepsize to compute ghost proposals later.

        Args:
            k: number of delayed rejection proposals

        Returns:
            float: new stepsize for each leapfrog step
        """
        stepsize = self._stepsize(k)
        validate_stepsize(stepsize)
        self._stepsize_list.append(stepsize)
        return stepsize

    def get_steps(self, k) -> int:
        """Computes new number of leapfrog steps for delayed rejection proposal.

        Stores new stepsize to compute ghost proposals later.

        Args:
            k: number of delayed rejection proposals

        Returns:
            int: new number of steps for leapfrog integrator
        """
        steps = self._steps(k)
        validate_steps(steps)
        self._steps_list.append(steps)
        return steps

    def sample(self) -> Draw:
        """Sample from target distribution with DrHmcDiag sampler.

        From current sample (theta, rho), propose new sample (theta_prop, rho_prop)
        and compute its acceptance probability. If accepted, return the new sample and
        its log density. If rejected, propose a new sample and repeat the process; stop
        when the maximum number of proposals is reached or when probabilistic delayed
        rejection determines not to propose a new sample.

        Flip momemntum after each proposal to ensure detailed balance.

        Returns:
            Draw: Tuple of (sample, log density)
        """
        rho = self._rng.normal(size=self._dim)
        logp = self.joint_logp(self._theta, rho)
        log_denom = 0
        for k in range(self._num_proposals):
            stepsize, steps = self.get_stepsize(k), self.get_steps(k)
            theta_prop, rho_prop = self.leapfrog(self._theta, rho, stepsize, steps)
            rho_prop *= -1
            accept_logp, logp_prop = self.accept(
                theta_prop, rho_prop, k, log_denom, logp
            )
            log_denom += 1 - np.exp(accept_logp)
            if np.log(self._rng.uniform()) < accept_logp:
                self._theta, rho = theta_prop, rho_prop
                logp = logp_prop
        return self._theta, logp

    def accept(
        self,
        theta_prop: NDArray[np.float64],
        rho_prop: NDArray[np.float64],
        k: int,
        log_denom: float,
        logp: float,
    ) -> tuple[float, float]:
        """Log acceptance probability of transitioning from current to proposed sample.

        Recursively compute log probability density of accepting transition from
        current sample (theta, rho) to proposed sample (theta_prop, rho_prop).
        Information about the current sample is contained in logp.

        Acceptance probability is computed following 'Delayed rejection Hamiltonian
        Monte Carlo for sampling multiscale distributions' by Chirag et al. In
        particular, we follow eqn. 29 (acceptance probability for k-th proposal of
        delayed rejection) eqn. 30 (acceptance probability for 2nd proposal of
        probablistic delayed rejection), and eqn. 31 (heuristic proposal
        probability) to compute the acceptance probability for the k-th probabilistic 
        delayed rejection proposal.

        Optimize computation by passing log_denom and logp from previous proposal
        and by returning logp_prop. This is less readable but ensures (1) only one
        density evaluation per sample, (2) no recursion is required to compute the
        denominator of eqn. 29, and (3) recursion on the numerator of eqn. 29 is done
        efficiently.

        Proposed samples are generated by running the leapfrog integrator from the
        *current sample*. To maintain detailed balance, 'ghost proposals' are similarly
        generated by running the leapfrog integrator from the *proposed sample*.
        (Proposed samples and ghost samples of the same order are generated with the
        same leapfrog stepsize and number of steps.) Eqn. 29 computes rejecting all
        previous proposed samples in the denominator and rejecting all previous ghost
        samples in the numerator. More details in section 3 of Chirag et al.

        Flip momemntum after each ghost proposal to ensure detailed balance.

        Args:
            theta_prop: proposed position
            rho_prop: proposed momentum
            k: number of delayed rejection proposals
            log_denom: log probability of rejecting all previous proposals
            logp: log probability of current sample

        Returns:
            tuple[float, float]: log probability of acceptance and proposed sample
        """
        logp_prop = self.joint_logp(theta_prop, rho_prop)
        log_num = 0
        for i in range(k):
            stepsize, steps = self._stepsize_list[i], self._steps_list[i]
            theta_ghost, rho_ghost = self.leapfrog(
                theta_prop, rho_prop, stepsize, steps
            )
            rho_ghost = -rho_ghost
            accept_logp, _ = self.accept(theta_ghost, rho_ghost, i, log_num, logp_prop)
            reject_logp = np.log1p(-np.exp(accept_logp) + 1e-16)
            log_num += reject_logp
        return min(0, (logp_prop - logp) + (log_num - log_denom)), logp_prop
