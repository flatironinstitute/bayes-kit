# bayes-kit

`bayes-kit` is an open-source Python package for black-box Bayesian
inference with minimial dependencies for maximal flexiblity.

*Bayesian models* involve specifying a posterior log density for
parameters conditioned on data.  Bayesian models are typically
specified in terms of a prior distribution over parameters and a
sampling distribution which generates observed data conditioned on the
parameters.  Users are free to define models using arbitrary Python
code, such as

* direct implementation in [NumPy](https://numpy.org),
* Python's foreign function interface to Fortran, C++, etc., or
* automatic differentiation libraries such as
    * [Stan](https://github.com/roualdes/bridgestan),
    * [PyTorch](https://pytorch.org),
	* [JAX](https://github.com/google/jax), or
    * [TensorFlow Probability](https://www.tensorflow.org/probability).

*Bayesian inference* involves conditioning on data and averaging over
uncertainty.  Specifically, `bayes-kit` can compute
    * parameter estimates,
    * posterior predictions for new observations, and
    * event probability forecasts,
all with Bayesian uncertainty quantification.

*Black-box* algorithms are agnostic to model structure.  Algorithms in
`bayes-kit` may require only log densities, whereas the
high-performance algorithms further require gradients of the log
density function.


## Markov chain Monte Carlo samplers

Markov chain Monte Carlo (MCMC) samplers provide a sequence of random
draws from target log density, which may be used for Monte Carlo
estimates of posterior expectations and quantiles for uncertainty
quantification.

#### Hamiltonian Monte Carlo sampler

Hamiltonian Monte Carlo simulates Hamiltonian dynamics with a
potential energy function equal to the negative log density. It
requires a target log density, gradient function, and optionally a
metric.

#### Random-walk Metropolis sampler

Random-walk Metropolis is a diffusive sampler that requires a
target log density function and a symmetric pseudorandom proposal
generator.


## Dependencies

`bayes-kit` has minimal dependencies, requiring only

* [NumPy](https://numpy.org).


## Licensing

* **Code** in this repository is released under the [MIT License](LICENSE-CODE).
* **Documentation** in this repository is released under the [CC BY 4.0 License](LICENSE-DOC).
