from typing import Iterator, Optional

import numpy as np

from .typing import DrawAndLogP, GradModel, Seed, VectorType


class DrGhmcDiag:
    """Generalized HMC sampler with Probabilistic Delayed Rejection diagonal metric.

    We seek to sample draws -- denoted by (theta, rho) -- from a target distribution
    specified in the model. As an HMC-variant, theta represents position (model
    parameters) and rho represents momentum.

    To produce a new draw, we generate a proposed draw with Hamiltonian dynamics. If
    accepted, the proposed draw and its log density are returned; if rejected, a new
    proposal is generated. This process is repeated until a maximum number of proposals
    is reached or when probabilistic delayed rejection returns early.

    This sampler requires non-increasing leapfrog stepsizes such that subsequent
    proposals are generated with (equal or) more accurate Hamiltonian dynamics. This
    allows for excellent sampling from multiscale distributions: large stepsizes are
    used in wide, flat regions while smaller stepsizes are used in narrow,
    high-curvature regions.

    Implementation based on Modi C, Barnett A, Carpenter B. Delayed rejection
    Hamiltonian Monte Carlo for sampling multiscale distributions. Bayesian Analysis.
    2023. https://doi.org/10.1214/23-BA1360.
    """

    def __init__(
        self,
        model: GradModel,
        proposals: int,
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
            model: model with log density and gradient
            proposals: maximum number of proposal attempts
            leapfrog_stepsizes: list of non-increasing leapfrog stepsizes
            leapfrog_stepcounts: list of number of leapfrog steps
            damping: Generalized HMC momentum damping factor in (0, 1]
            metric_diag: diagonal of the metric. Defaults to identity metric.
            init: initialize Markov chain. Defaults to draw from standard normal.
            seed: seed for numpy rng. Defaults to non-reproducible rng.
            prob_retry: retry another proposal probabilistically. Defaults to True.
        """
        self._model = model
        self._dim = self._model.dims()
        self._proposals = self._validate_propoals(proposals)
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

        # When generating a draw, the cache is used in the joint_logp() and leapfrog()
        # functions to avoid recomputing log densities and gradients of the same draw.
        # Across draws, cache usage depends on the the proposed draw: if accepted,
        # the cache retains the log density and gradient of the proposed draw; if
        # rejected, the cache retains the log density and gradient of the current draw.
        self._cache: list = []

    def _validate_propoals(self, proposals: int) -> int:
        """Validate number of proposals.

        Args:
            proposals: maximum number of proposal attempts

        Raises:
            TypeError: proposals is not an int
            ValueError: proposals is less than one

        Returns:
            validated number of proposals
        """
        if not (type(proposals) is int):
            raise TypeError(f"proposals must be an int, not {type(proposals)}")
        if not (proposals >= 1):
            raise ValueError(
                f"proposals must be greater than or equal to 1, not {proposals}"
            )
        return proposals

    def _validate_damping(self, damping: float) -> float:
        """Validate damping factor.

        Args:
            damping: Generalized HMC momentum damping factor in (0, 1]

        Raises:
            TypeError: damping is not a float
            ValueError: damping is not in (0, 1]

        Returns:
            validated damping factor
        """
        if not (type(damping) is float):
            raise TypeError(f"damping must be a float, not {type(damping)}")
        if not 0 < damping <= 1:
            raise ValueError(f"damping of {damping} must be within (0, 1]")
        return damping

    def _validate_leapfrog_stepsizes(
        self, leapfrog_stepsizes: list[float]
    ) -> list[float]:
        """Validate list of leapfrog stepsizes.

        Args:
            leapfrog_stepsizes: list of non-increasing leapfrog stepsizes

        Raises:
            TypeError: leapfrog stepsizes is not a list
            ValueError: leapfrog stepsize list is of incorrect length
            TypeError: leapfrog stepsize list contains non-float stepsizes
            ValueError: leapfrog stepsize list contains non-positive stepsizes

        Returns:
            list of validated leapfrog stepsizes
        """
        if not (type(leapfrog_stepsizes) is list):
            raise TypeError(
                f"leapfrog_stepsizes must be a list, not {type(leapfrog_stepsizes)}"
            )
        if len(leapfrog_stepsizes) != self._proposals:
            raise ValueError(
                f"leapfrog_stepsizes must be a list of length {self._proposals}, not"
                f" length {len(leapfrog_stepsizes)}, so that each proposal has a "
                "specified leapfrog stepsize"
            )
        for idx, stepsize in enumerate(leapfrog_stepsizes):
            if not (type(stepsize) is float):
                raise TypeError(
                    f"leapfrog stepsizes must be of type float, not {type(stepsize)} "
                    f"at index {idx}"
                )
            if not stepsize > 0:
                raise ValueError(
                    f"leapfrog stepsizes must be positive, but found stepsize of "
                    f"{stepsize} at index {idx}"
                )
        for idx, (prev, cur) in enumerate(
            zip(leapfrog_stepsizes[:-1], leapfrog_stepsizes[1:])
        ):
            if not cur <= prev:
                raise ValueError(
                    f"leapfrog stepsizes must be non-increasing, but found stepsize of "
                    f"{cur} at index {idx + 1} which is greater than stepsize of "
                    f"{prev} at index {idx}"
                )
        return leapfrog_stepsizes

    def _validate_leapfrog_stepcounts(
        self, leapfrog_stepcounts: list[int]
    ) -> list[int]:
        """Validate list of leapfrog stepcounts.

        Args:
            leapfrog_stepcounts: list of leapfrog stepcounts

        Raises:
            TypeError: leapfrog stepcounts is not a list
            ValueError: leapfrog stepcount list is of incorrect length
            TypeError: leapfrog stepcounts list contains non-integer steps
            ValueError: leapfrog stepcounts list contains non-positive steps

        Returns:
            list of validated leapfrog stepcounts
        """
        if not (type(leapfrog_stepcounts) is list):
            raise TypeError(
                f"leapfrog_stepcounts must be a list, not {type(leapfrog_stepcounts)}"
            )
        if len(leapfrog_stepcounts) != self._proposals:
            raise ValueError(
                f"leapfrog_stepcounts must be a list of length {self._proposals}, not"
                f" length {len(leapfrog_stepcounts)}, so that each proposal has a "
                "specified number of leapfrog steps"
            )
        for idx, stepcount in enumerate(leapfrog_stepcounts):
            if not (type(stepcount) is int):
                raise TypeError(
                    f"leapfrog stepcounts must be of type int, not {type(stepcount)} "
                    f"at index {idx}"
                )
            if not stepcount > 0:
                raise ValueError(
                    f"leapfrog stepcounts must be positive, but found stepcount of "
                    f"{stepcount} at index {idx}"
                )
        return leapfrog_stepcounts

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
        """Log density of draw (theta, rho) under the unnormalized Gibbs pdf.

        The Gibbs distribution, also known as the canonical or Boltzmann distribution,
        is a probability density function (pdf) that depends on the Hamiltonian. The
        Hamiltonian is the sum of the potential and kinetic energies, defined by the
        position and momentum respectively.

        Args:
            theta: position (model parameters)
            rho: momentum

        Returns:
            unnormalized, joint log density of draw (theta, rho)
        """

        if not self._cache:
            logp, grad = self._model.log_density_gradient(theta)
            self._cache = [(logp, grad)]
        else:
            logp, _ = self._cache[-1]

        potential = -logp
        kinetic = 0.5 * np.dot(rho, self._metric * rho)
        hamiltonian = potential + kinetic
        return -hamiltonian

    def leapfrog(
        self,
        theta: VectorType,
        rho: VectorType,
        stepsize: float,
        stepcount: int,
    ) -> tuple[VectorType, VectorType]:
        """Simulate Hamiltonian dynamics by leapfrog integration for draw (theta, rho).

        Discretize and solve Hamilton's equations using the leapfrog integrator with
        specified stepsize and stepcount.

        When leapfrog integration is followed by a momentum flip, this generates a new,
        proposed draw (theta_prop, rho_prop) from the current draw (theta, rho).

        Args:
            theta: position (model parameters)
            rho: momentum
            stepsize: stepsize in each leapfrog step
            stepcount: number of leapfrog steps

        Returns:
            Hamiltonian dynamics simulated for draw (theta, rho)
        """
        theta = np.array(theta, copy=True)  # copy b/c numpy's += mutates original array

        logp, grad = self._cache[-1]
        rho_mid = rho + 0.5 * stepsize * np.multiply(self._metric, grad).squeeze()
        theta += stepsize * rho_mid

        for _ in range(stepcount - 1):
            logp, grad = self._model.log_density_gradient(theta)
            rho_mid += stepsize * np.multiply(self._metric, grad).squeeze()
            theta += stepsize * rho_mid

        logp, grad = self._model.log_density_gradient(theta)
        rho = rho_mid + 0.5 * stepsize * np.multiply(self._metric, grad).squeeze()
        self._cache.append((logp, grad))
        return (theta, rho)

    def retry_logp(self, reject_logp: float) -> float:
        """Log probability of attempting, or retrying, another proposal upon rejection.

        To reduce average cost per iteration, make the delayed rejections
        probabilistic, such that a subsequent proposal is not mandatory upon rejection.
        Instead, the probability of making proposal k+1, after rejecting proposals
        1...k, is the probability of rejecting proposals 1...k.

        The probability of attempting another proposal depends on where we are in the
        distribution. If the previous proposal has low acceptance probability, propose
        another draw; if the previous proposal has high acceptance probablity but was
        rejected by chance, do not propose another draw. This avoid unnecessary,
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
        """Map current draw (theta, rho) to proposed draw (theta_prop, rho_prop).

        The proposal map generates a proposed draw (theta_prop, rho_prop) from the
        current draw (theta, rho). The proposed draw is computed as a deterministic
        composition of the leapfrog integrator and a momentum flip.

        Because this proposal map is a deterministic volume-preserving involution, it
        maintains detailed balance as per Modi et al. (2023).

        Args:
            theta: position (model parameters)
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
        """Draw from the target distribution with this sampler.

        From current draw (theta, rho), propose new draw (theta_prop, rho_prop)
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

        for k in range(self._proposals):
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
            self._cache.pop()  # cache is set in proposal_map() -> leapfrog()

        self._cache = [self._cache.pop()]
        self._rho = -self._rho  # negate the momentum unconditionally for GHMC
        return self._theta, cur_logp

    def accept(
        self,
        theta_prop: VectorType,
        rho_prop: VectorType,
        k: int,
        cur_hastings: float,
        cur_logp: float,
    ) -> tuple[float, float]:
        """Log acceptance probability of transitioning from current to proposed draw.

        To maintain detailed balance, symmetrically consider (1) transitioning from
        the current draw (theta, rho) to the proposed draw (theta_prop, rho_prop) and
        (2) transitioning from the proposed draw back to the current draw.

        For each direction, compute three terms: the joint log density, the Hastings
        term, and the probability of retrying another proposal upon rejection.

        For direction (1), the Hastings term is the probability of rejecting all
        previous proposals (theta_prop, rho_prop), generated by applying previous
        proposal maps to the *current draw*.

        For direction (2), the Hastings term is the probability of rejecting all ghost
        proposals (theta_ghost, rho_ghost), generated by applying the same proposal
        maps to the *proposed draw*.

        Ghost draws represent proposals that would have been made in a hypothetical
        chain, had we started the chain in the reverse direction. While ghost draws are
        never proposed, maintaining detailed balance requires evaluating the joint
        Gibbs distribution at these points.

        The log acceptance probability computed in this function extends eqns. 29, 30,
        and 31 of Modi et al. (2023).

        Args:
            theta_prop: proposed position (proposed model parameters)
            rho_prop: proposed momentum
            k: proposal number (for leapfrog stepsize and stepcount)
            cur_hastings: log probability of rejecting all previous proposals
            cur_logp: log probability of current draw

        Returns:
            log probability of acceptance and proposed draw
        """
        # Optimize computation by storing information about the current draw in
        # cur_logp and *not* passing it into the function directly via (theta, rho).

        # Optimize computation by passing cur_hastings and cur_logp from previous proposal
        # and by returning prop_logp. This is less readable but ensures (1) only one
        # density evaluation per draw outside of leapfrog, (2) no recursion is required
        # to compute the denominator of eqn. 29, and (3) recursion on the numerator of
        # eqn. 29 is done efficiently -- for equations from Modi et al. (2023).

        prop_logp = self.joint_logp(theta_prop, rho_prop)
        prop_hastings = 0

        for i in range(k):
            theta_ghost, rho_ghost = self.proposal_map(theta_prop, rho_prop, i)
            accept_logp, _ = self.accept(
                theta_ghost, rho_ghost, i, prop_hastings, prop_logp
            )

            if accept_logp == 0:  # early stopping to avoid -inf in np.log1p
                self._cache.pop()  # cache is set in proposal_map() -> leapfrog()
                return -np.inf, prop_logp

            reject_logp = np.log1p(-np.exp(accept_logp))
            prop_hastings += reject_logp
            self._cache.pop()  # cache is set in proposal_map() -> leapfrog()

        prop_retry_logp = self.retry_logp(prop_hastings)
        cur_retry_logp = self.retry_logp(cur_hastings)

        accept_frac = (
            (prop_logp - cur_logp)
            + (prop_hastings - cur_hastings)
            + (prop_retry_logp - cur_retry_logp)
        )
        return min(0, accept_frac), prop_logp
