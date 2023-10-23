from typing import Iterator, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .typing import DrawAndLogP, GradModel, Seed, VectorType


class DrGhmcDiag:
    """(Probabilistic) Delayed Rejection Generalized Hamiltonian Monte Carlo sampler.
    
    Sample from a target probability distribution by generating a sequence of draws. 
    Each draw (theta, rho) consists of position variable theta and auxiliary momentum 
    variable rho, with theta representing a single sample of the target distribution's 
    parameters.

    To produce a new draw, partially refresh the momentum and generate a proposed draw
    with Hamiltonian dynamics. If accepted, the proposed draw and its log density are
    returned. If rejected, compute the probability of retrying another proposal, based
    on the acceptance probability of the rejected draw; if a fresh uniform(0, 1) 
    variate is less than this probability, generate a new proposed draw. This process 
    is repeated until a maximum number of proposals is reached.

    This efficiently samples from multiscale distributions because of probabilistic 
    delayed rejection and partial momentum refresh. With non-increasing leapfrog 
    stepsizes, the former encourages large stepsizes in wide, flat regions and smaller 
    stepsizes in narrow, high-curvature regions. The latter suppresses random-walk 
    behavior by only partially updating the auxiliary momentum variable rho.

    This implementation is based on Modi C, Barnett A, Carpenter B. Delayed rejection 
    Hamiltonian Monte Carlo for sampling multiscale distributions. Bayesian Analysis.
    2023. https://doi.org/10.1214/23-BA1360.
    """

    def __init__(
        self,
        model: GradModel,
        max_proposals: int,
        leapfrog_stepsizes: list[float],
        leapfrog_stepcounts: list[int],
        damping: float,
        metric_diag: Optional[VectorType] = None,
        init: Optional[VectorType] = None,
        seed: Optional[Seed] = None,
        prob_retry: bool = True,
    ):
        """Initialize the DrGhmcDiag sampler.

        Args:
            model: probabilistic model with log density and gradient
            max_proposals: maximum number of proposal attempts
            leapfrog_stepsizes: list of leapfrog stepsizes
            leapfrog_stepcounts: list of number of leapfrog steps
            damping: generalized HMC momentum damping factor in (0, 1]
            metric_diag: diagonal of a diagonal metric. Defaults to identity metric.
            init: parameter vector to initialize position variable theta. Defaults to 
                draw from standard normal.
            seed: seed for Numpy RNG. Defaults to non-reproducible RNG.
            prob_retry: boolean flag for using probabilistic delayed rejection. 
                Defaults to True.
        """
        self._model = model
        self._dim = self._model.dims()
        self._max_proposals = self._validate_propoals(max_proposals)
        self._leapfrog_stepsizes = self._validate_leapfrog_stepsizes(leapfrog_stepsizes)
        self._leapfrog_stepcounts = self._validate_leapfrog_stepcounts(
            leapfrog_stepcounts
        )
        self._damping = self._validate_damping(damping)
        self._metric = metric_diag or np.ones(self._dim)
        self._rng = np.random.default_rng(seed)
        self._theta = (
            init
            if (init is not None and init.shape != (0,))
            else self._rng.normal(size=self._dim)
        )
        self._rho = self._rng.normal(size=self._dim)
        self._prob_retry = prob_retry

        # use stack to avoid redundant computation within a single draw (when 
        # recursively computing the log acceptance probability) and across draws
        self._log_density_gradient_cache: list[Tuple[float, VectorType]] = []

    def _validate_propoals(self, max_proposals: int) -> int:
        """Check that the maximum number of proposals is an integer greater than or 
        equal to one. 

        Args:
            max_proposals: maximum number of proposal attempts

        Raises:
            TypeError: max_proposals is not an int
            ValueError: max_proposals is less than one

        Returns:
            validated number of proposals
        """
        if not (type(max_proposals) is int):
            raise TypeError(f"max_proposals must be an int, not {type(max_proposals)}")
        if not (max_proposals >= 1):
            raise ValueError(
                f"max_proposals must be greater than or equal to 1, not {max_proposals}"
            )
        return max_proposals

    def _validate_leapfrog_stepsizes(
        self, leapfrog_stepsizes: list[float]
    ) -> list[float]:
        """Check that leapfrog stepsizes is a list with positive, float stepsizes and a
        length equal to the maximum number of proposals.

        Args:
            leapfrog_stepsizes: list of leapfrog stepsizes

        Raises:
            TypeError: leapfrog_stepsizes is not a list
            ValueError: leapfrog_stepsizes is of incorrect length
            TypeError: leapfrog_stepsizes contains non-float stepsizes
            ValueError: leapfrog_stepsizes contains non-positive stepsizes

        Returns:
            list of validated leapfrog stepsizes
        """
        if not (type(leapfrog_stepsizes) is list):
            raise TypeError(
                f"leapfrog_stepsizes must be of type list, but found type "
                f"{type(leapfrog_stepsizes)}"
            )
        if len(leapfrog_stepsizes) != self._max_proposals:
            raise ValueError(
                f"leapfrog_stepsizes must be a list of length {self._max_proposals} "
                f"so that each proposal has its own specfied leapfrog stepsize, but "
                f"instead found length {len(leapfrog_stepsizes)}"
            )
        for idx, stepsize in enumerate(leapfrog_stepsizes):
            if not (type(stepsize) is float):
                raise TypeError(
                    f"each stepsize in leapfrog_stepsizes must be of type float, but "
                    f"found stepsize of type {type(stepsize)} at index {idx}"
                )
            if not stepsize > 0:
                raise ValueError(
                    f"each stepsize in leapfrog_stepsizes must be positive, but found "
                    f"stepsize of {stepsize} at index {idx}"
                )
        return leapfrog_stepsizes

    def _validate_leapfrog_stepcounts(
        self, leapfrog_stepcounts: list[int]
    ) -> list[int]:
        """Check that leapfrog stepcounts is a list with positive, integer stepcounts 
        and a length equal to the maximum number of proposals.

        Args:
            leapfrog_stepcounts: list of leapfrog stepcounts

        Raises:
            TypeError: leapfrog_stepcounts is not a list
            ValueError: leapfrog_stepcounts is of incorrect length
            TypeError: leapfrog_stepcounts contains non-integer steps
            ValueError: leapfrog_stepcounts contains non-positive steps

        Returns:
            list of validated leapfrog stepcounts
        """
        if not (type(leapfrog_stepcounts) is list):
            raise TypeError(
                f"leapfrog_stepcounts must be of type list, but found type "
                f"{type(leapfrog_stepcounts)}"
            )
        if len(leapfrog_stepcounts) != self._max_proposals:
            raise ValueError(
                f"leapfrog_stepcounts must be a list of length {self._max_proposals}, "
                f"so that each proposal has its own specified number of leapfrog "
                f"steps, but instead found length {len(leapfrog_stepcounts)}"
            )
        for idx, stepcount in enumerate(leapfrog_stepcounts):
            if not (type(stepcount) is int):
                raise TypeError(
                    f"each stepcount in leapfrog_stepcounts must be of type int, but "
                    f"found stepcount of type {type(stepcount)} at index {idx}"
                )
            if not stepcount > 0:
                raise ValueError(
                    f"each stepcount in leapfrog_stepcounts must be positive, but "
                    f"found stepcount of {stepcount} at index {idx}"
                )
        return leapfrog_stepcounts

    def _validate_damping(self, damping: float) -> float:
        """Check that the damping factor is a float in (0, 1].

        Args:
            damping: generalized HMC momentum damping factor in (0, 1]

        Raises:
            TypeError: damping is not a float
            ValueError: damping is not in (0, 1]

        Returns:
            validated damping factor
        """
        if not (type(damping) is float):
            raise TypeError(
                f"damping must be of type float, but found type {type(damping)}"
            )
        if not 0 < damping <= 1:
            raise ValueError(
                f"damping must be within (0, 1], but found damping of {damping}"
            )
        return damping

    def __iter__(self) -> Iterator[DrawAndLogP]:
        """Return the iterator for draws from this sampler.

        Returns:
            Iterator of draws from DrGhmcDiag sampler
        """
        return self

    def __next__(self) -> DrawAndLogP:
        """Return the next draw from this sampler, along with its log density.

        Yields:
            Tuple of (draw, log density)
        """
        return self.sample()

    def joint_logp(self, theta: VectorType, rho: VectorType) -> float:
        """Return the log density of draw (theta, rho) for the unnormalized Gibbs pdf.

        The Gibbs distribution, also known as the canonical or Boltzmann distribution,
        is a probability density function (pdf) that depends on the Hamiltonian. The
        Hamiltonian is the sum of the potential and kinetic energies, defined by the
        position and momentum respectively.

        Assume that the metric is diagonal.

        Args:
            theta: position
            rho: momentum

        Returns:
            unnormalized, joint log density of draw (theta, rho)
        """

        if not self._log_density_gradient_cache:
            logp, grad = self._model.log_density_gradient(theta)
            self._log_density_gradient_cache = [(logp, np.asanyarray(grad))]
        else:
            logp, _ = self._log_density_gradient_cache[-1]

        potential = -logp
        kinetic: float = 0.5 * np.dot(rho, self._metric * rho)
        return -(potential + kinetic)

    def leapfrog(
        self,
        theta: VectorType,
        rho: VectorType,
        stepsize: float,
        stepcount: int,
    ) -> tuple[VectorType, VectorType]:
        """Return the result of running the leapfrog integrator for Hamiltonian 
        dynamics starting from the current draw (theta, rho) with the specified step 
        size and number of steps.

        Args:
            theta: position
            rho: momentum
            stepsize: stepsize in each leapfrog step
            stepcount: number of leapfrog steps

        Returns:
            Approximate solution to Hamiltonian dynamics via leapfrog integration
        """
        theta = np.array(theta, copy=True)  # copy so as not to mutate theta
        grad: ArrayLike  # mypy infers too strict a type when reading from cache

        logp, grad = self._log_density_gradient_cache[-1]
        rho_mid = rho + 0.5 * stepsize * np.multiply(self._metric, grad).squeeze()
        theta += stepsize * rho_mid

        for _ in range(stepcount - 1):
            logp, grad = self._model.log_density_gradient(theta)
            rho_mid += stepsize * np.multiply(self._metric, grad).squeeze()
            theta += stepsize * rho_mid

        logp, grad = self._model.log_density_gradient(theta)
        rho = rho_mid + 0.5 * stepsize * np.multiply(self._metric, grad).squeeze()

        self._log_density_gradient_cache.append((logp, np.asanyarray(grad)))
        return (theta, rho)

    def retry_logp(self, reject_logp: float) -> float:
        """Return the log density of retrying another proposal upon rejection.

        To reduce average cost per iteration, make the delayed rejections
        probabilistic, such that a subsequent proposal is not mandatory upon rejection.
        Instead, the probability of making proposal k+1, after rejecting proposals
        1...k, is the probability of rejecting proposals 1...k.

        The probability of attempting another proposal depends on where we are in the
        distribution. If the previous proposal has low acceptance probability, propose
        another draw; if the previous proposal has high acceptance probablity but was
        rejected by chance, do not propose another draw. This avoids unnecessary,
        additional proposals in high-density regions.

        To maintain detailed balance, the acceptance probability is multiplied by an
        extra term, computed in this function: the Hastings term of the propopsed draw
        divided by the Hastings term of the current draw. More details can be found
        in section 3.2 of Modi et al. (2023).

        Args:
            reject_logp: log probability of rejecting all previous proposals

        Returns:
            log probability of making another proposal
        """
        # equivalent to: reject_logp if self._prob_retry else 0.0
        return self._prob_retry * reject_logp

    def proposal_map(
        self, theta: VectorType, rho: VectorType, k: int
    ) -> tuple[VectorType, VectorType]:
        """Return the proposed draw starting at the specified position and momentum 
        given the proposal number.

        The proposal map generates a proposed draw (theta_prop, rho_prop) from the
        current draw (theta, rho). The proposed draw is computed as a deterministic
        composition of the leapfrog integrator and a momentum flip.

        Because this proposal map is a deterministic symplectic involution, it
        maintains detailed balance as per Modi et al. (2023).

        Args:
            theta: position
            rho: momentum
            k: proposal number (for leapfrog stepsize and stepcount)

        Returns:
            proposed draw (theta_prop, rho_prop)
        """
        stepsize, stepcount = self._leapfrog_stepsizes[k], self._leapfrog_stepcounts[k]
        theta_prop, rho_prop = self.leapfrog(theta, rho, stepsize, stepcount)
        rho_prop = -rho_prop
        return (theta_prop, rho_prop)

    def sample(self) -> DrawAndLogP:
        """Return the next draw in the Markov chain defined by this class.

        From the current draw (theta, rho), propose a new draw (theta_prop, rho_prop) 
        and compute its acceptance probability. If accepted, return the new draw and 
        its log density; if rejected, propose a new draw and repeat. Stop when the 
        maximum number of proposals is reached or when probabilistic delayed rejection 
        returns early.

        Returns:
            Tuple of (draw, log density)
        """
        self._rho = self._rng.normal(
            loc=self._rho * np.sqrt(1 - self._damping),
            scale=np.sqrt(self._damping),
            size=self._dim,
        )
        cur_logp = self.joint_logp(self._theta, self._rho)
        cur_hastings, reject_logp = 0.0, 0.0

        for k in range(self._max_proposals):
            retry_logp = self.retry_logp(reject_logp)
            if not np.log(self._rng.uniform()) < retry_logp:
                break

            theta_prop, rho_prop = self.proposal_map(self._theta, self._rho, k)
            accept_logp, prop_logp = self.accept(
                theta_prop, rho_prop, k, cur_hastings, cur_logp
            )

            if np.log(self._rng.uniform()) < accept_logp:
                self._theta, self._rho = theta_prop, rho_prop
                cur_logp = prop_logp
                break

            reject_logp = np.log1p(-np.exp(accept_logp))
            cur_hastings += reject_logp
            self._log_density_gradient_cache.pop()  # cache set in leapfrog() function

        self._log_density_gradient_cache = [self._log_density_gradient_cache.pop()]
        self._rho = -self._rho  # negate momentum unconditionally for generalized HMC
        return self._theta, cur_logp

    def accept(
        self,
        theta_prop: VectorType,
        rho_prop: VectorType,
        k: int,
        cur_hastings: float,
        cur_logp: float,
    ) -> tuple[float, float]:
        """ Return a tuple containing the log acceptance probability and proposed draw.

        Calculate the log acceptance probability of transitioning from the current draw 
        (theta, rho) to the proposed draw (theta_prop, rho_prop) by computing the ratio
        of three terms in the forward and reverse directions: the joint log density, 
        the Hastings factor, and the probability of retrying another proposal upon 
        rejection.

        This function extends equations 29, 30, and 31 of Modi et al. (2023) by 
        computing the log acceptance probability for *any* number of proposals with 
        probabilistic delayed rejection. See Modi et al. (2023) for details.

        Args:
            theta_prop: proposed position
            rho_prop: proposed momentum
            k: proposal number (for leapfrog stepsize and stepcount)
            cur_hastings: log probability of rejecting all previous proposals
            cur_logp: log probability of current draw

        Returns:
            log probability of acceptance and proposed draw
        """
        prop_logp = self.joint_logp(theta_prop, rho_prop)
        prop_hastings = 0

        for i in range(k):
            theta_ghost, rho_ghost = self.proposal_map(theta_prop, rho_prop, i)
            accept_logp, _ = self.accept(
                theta_ghost, rho_ghost, i, prop_hastings, prop_logp
            )

            if accept_logp == 0:  # early stopping to avoid -inf in np.log1p
                self._log_density_gradient_cache.pop()  # cache set in leapfrog() function
                return -np.inf, prop_logp

            reject_logp = np.log1p(-np.exp(accept_logp))
            prop_hastings += reject_logp
            self._log_density_gradient_cache.pop()  # cache set in leapfrog() function

        prop_retry_logp = self.retry_logp(prop_hastings)
        cur_retry_logp = self.retry_logp(cur_hastings)

        accept_frac = (
            (prop_logp - cur_logp)
            + (prop_hastings - cur_hastings)
            + (prop_retry_logp - cur_retry_logp)
        )
        return min(0, accept_frac), prop_logp
