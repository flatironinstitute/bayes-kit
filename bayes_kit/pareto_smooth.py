import numpy as np

def generalized_pareto_estimate(x):
    """Return the estimates of k and sigma in the generalized Pareto distribution for the specified values.

    The location parameter mu is set internally to `30 + sqrt(len(x))`.  

    The basic estimation algorithm is derived from this paper:
    
    J. Zhang, M.A. Stephens. 2009. A new and efficient estimation method
    for the generalized Pareto distribution. Technometrics 51(3).
    https://www.jstor.org/stable/40586625
    
    The code was ported from the arviz package: https://python.arviz.org/en/stable/_modules/arviz/stats/stats.html#loo

    Args:
        x: An array of positive data values.

    Returns:
        A pair (k_hat, sigma_hat) with estimates of parameters k and sigma
    """
    y = np.sort(x)
    n = len(y)
    m_hat = 30 + int(np.sqrt(n))
    b = 1 - np.sqrt(m_hat / (np.arange(m_hat, dtype=float) + 0.5))
    prior_bs = 3
    b = b / (prior_bs * y[(n - 2) // 4]) + 1 / y[-1]
    k = np.log1p(-b[:, None] * y).mean(axis=1)
    len_scale = n * (np.log(-(b / k)) - k - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    b_post = np.sum(b * weights) / weights.sum()
    k_post = np.log1p(-b_post * y).mean()
    sigma_hat = -k_post / b_post
    prior_k_count = 10
    prior_k_value = 0.5
    k_hat = (n * k_post + prior_k_count * prior_k_value) / (n + prior_k_count)
    return k_hat, sigma_hat


def generalized_pareto_quantile(u, mu, sigma, k):
    """Return the quantile u of the generalized Pareto distribution with location mu, scale sigma, and shape kappa.
    
    Args:
        u: A value in (0, 1) determining the quantile.
        mu: The location (minimum value) of the distribution.
        sigma: The scale of the distribution.
        k: The shape of the distribution.

    Return:
        The value of the generalized Pareto distribution for which the CDF is u.
    """
    return mu + sigma / k * ((1 - u)**(-k) - 1)


def pareto_smooth(weights_raw):
    """Return the result of Pareto smoothing the specified array of positive importance weights.

    A description of the algorithm can be found in: Vehtari, A.,
    Simpson, D., Gelman, A., Yao, Y. and Gabry, J., 2015. Pareto
    smoothed importance sampling. arXiv 1507.02646.
    
    Args:
    weights_raw:  The raw weights to smooth (positive values only).

    Return:
    A pair (weights, k_hat) of the smoothed weights and estimate of k for the distribution.
    """
    weights = np.array(weights_raw)
    S = len(weights)
    M = int(np.minimum(0.2 * S, 3 * np.sqrt(S)))
    max_weight = np.max(weights)
    idxs = np.argpartition(-weights, M)[:M]
    largest_weights = weights[idxs]
    kappa_hat, sigma_hat = generalized_pareto_estimate(largest_weights)
    ranks = np.argsort(np.argsort(largest_weights))
    mu_hat = np.sort(weights)[-M]
    weights[idxs] = [
        np.minimum(
            max_weight,
            generalized_pareto_quantile((rank + 0.5) / M, mu_hat, sigma_hat, kappa_hat),
        )
        for rank in ranks
    ]
    return weights, kappa_hat
