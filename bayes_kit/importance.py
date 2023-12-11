import numpy as np
import scipy as sp


def map_sample(f, draws):
    return np.apply_along_axis(f, arr=draws, axis=0)

def to_simplex(log_weights):
    return np.exp(log_weights - sp.special.logsumexp(log_weights))

def is_weights(proposal, target, num_draws):
    draws = [proposal.sample() for _ in range(num_draws)]
    log_p = map_sample(target.log_density, draws)
    log_q = map_sample(proposal.log_density, draws)
    weights = to_simplex(log_p - log_q)
    return draws, weights

def is_expect(draws, weights, f):
    return (map_sample(f, draws) * weights).sum(axis=-1)

def importance_sample(proposal, target, num_draws, f):
    draws, weights = is_weights(proposal, target, num_draws)
    return is_expect(draws, weights, f)

def importance_resample(proposal, target, pool_size, num_draws):
    pool, weights = is_weights(proposal, target, pool_size)
    counts = np.random.multinomial(num_draws, weights)
    return np.repeat(pool, counts, axis=0)




def generalized_pareto_estimate(x):
    # A. Vehtari, D. Simpson, A. Gelman, Y. Yao, J. Gabry. 2022. Pareto smoothed
    # importance sampling.  arXiv:1507.02646. https://arxiv.org/pdf/1507.02646.pdf
    # J. Zhang, M.A. Stephens. 2009. A new and efficient estimation method for
    # the generalized Pareto distribution. Technometrics 51(3).
    # https://www.jstor.org/stable/40586625
    theta_hat = 1
    kappa_hat = 1
    return theta_hat, kappa_hat


def pareto_smooth(weights):
    kappa = 0.5
    return weights, kappa


def ps_is_weights(proposal, target, num_draws):
    draws, weights = is_weights(proposal, target, num_draws)
    smoothed_weights, kappa = pareto_smooth(weights)
    return draws, smoothed_weights, kappa


def ps_is_expect(draws, weights, f):
    smoothed_weights, kappa = pareto_smooth(weights)
    return is_expect(draws, smoothed_weights, f), kappa


def ps_importance_sample(proposal, target, num_draws, f):
    kappa = 0.5
    return importance_sample(proposal, target, num_draws, f), kappa


def importance_resample_ps(proposal, target, pool_size, num_draws):
    kappa = 0.5
    return importance_resample(proposal, target, pool_size, num_draws)
