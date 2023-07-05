# bayes-kit

`bayes-kit` is an open-source Python package for Bayesian inference
and posterior analysis with minimial dependencies for maximal
flexiblity.

## Example

The following example defines a model `StdNormal`, samples 1000 draws
using Metropolis-adjusted Langevin sampling, and prints the mean and
variance estimates.

```python
import numpy as np
from bayes_kit.mala import MALA

class StdNormal:
	def dims(self):
		return 1
	def log_density(self, theta):
		return -0.5 * theta[0]**2
	def log_density_gradient(self, theta):
		return self.log_density(theta), -theta
	
model = StdNormal()
sampler = MALA(model, 0.2)
M = 1000
draws = np.array([sampler.sample()[0] for _ in range(M)])

print(f"{draws.mean()=}  {draws.var()=}")
print(f"{draws[0:5]=}")
```

## Model specification

For `bayes-kit`, a *Bayesian model* is specified as a class
implementing a log density function with optional gradients and
Hessians.  Models may be defined using arbitrary Python code,
including 

* direct implementation in [NumPy](https://numpy.org),
* Python's foreign function interface to Fortran, C++, etc., or
* automatic differentiation libraries such as
    * [Stan](https://github.com/roualdes/bridgestan),
    * [PyTorch](https://pytorch.org),
	* [JAX](https://github.com/google/jax), or
    * [TensorFlow Probability](https://www.tensorflow.org/probability).

## Bayesian inference

*Bayesian inference* involves conditioning on observed data and
averaging over uncertainty.  Specifically, `bayes-kit` can compute
    * parameter estimates,
    * posterior predictions for new observations, and
    * event probability forecasts,
all with Bayesian uncertainty quantification.

Algorithms in `bayes-kit` rely only on a model's log density function
and optionally its derivatives, not on any particular model structure.
As such, they are closed-box algorithms that treat models as
encapsulated.

High performance algorithms that scale well with dimension require
gradients and algorithms that adapt to varying curvature require
Hessians. 

### Markov chain Monte Carlo samplers

Markov chain Monte Carlo (MCMC) samplers provide a sequence of random
draws from target log density, which may be used for Monte Carlo
estimates of posterior expectations and quantiles for uncertainty
quantification.

#### Random-walk Metropolis sampler

Random-walk Metropolis (RWM) is a diffusive sampler that requires a
target log density function and a symmetric pseudorandom proposal
generator.

#### Metropolis-adjusted Langevin sampler

Metroplis-adjusted Langevin (MALA) is a diffusive sampler that adjusts
proposals with gradient-based information.  MALA requires a target log
density and gradient function.

#### Hamiltonian Monte Carlo sampler

Hamiltonian Monte Carlo (HMC) simulates Hamiltonian dynamics with a
potential energy function equal to the negative log density. It
requires a target log density, gradient function, and optionally a
metric.

### Sequential Monte Carlo samplers

Sequential Monte Carlo (SMC) samplers alternate proposing moves and
sampling importance resampling for a sequence of intermediate
densities that end in the target density.

#### Likelihood annealed sequential Monte Carlo sampler

In this approach to SMC, the target at step `n` is proportional to

```
p(theta | y)^t[n] * p(theta),
```

where the temperature `t[n]` runs from 0 to 1 across iterations.


## Dependencies

`bayes-kit` only depends on two external packages,

* [NumPy](https://numpy.org), and
* [SciPy](https://scipy.org).

## Licensing

* **Code** in this repository is released under the [MIT License](LICENSE-CODE).
* **Documentation** in this repository is released under the [CC BY 4.0 License](LICENSE-DOC).
