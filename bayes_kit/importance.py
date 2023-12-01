import numpy as np
import scipy as sp

def is_weights(proposal, target, num_draws):
    draws = [proposal.sample() for _ in range(num_draws)]
    log_weights = [target.log_density(draw) - proposal.log_density(draw) for draw in draws]
    weights = np.exp(log_weights - sp.special.logsumexp(log_weights))
    return draws, weights

def is_expect(draws, weights, f):
    return np.sum([weight * f(draw) for draw, weight in zip(draws, weights)], axis=0)

def importance_sample(proposal, target, num_draws, f):
    draws, weights = is_weights(proposal, target, num_draws)
    return is_expect(draws, weights, f)

def importance_resample(proposal, target, pool_size, num_draws):
    pool, weights = is_weights(proposal, target, pool_size)
    counts = np.random.multinomial(num_draws, weights)
    draws = []
    for draw, count in zip(pool, counts):
        draws.extend([draw] * count)
    return draws        

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
