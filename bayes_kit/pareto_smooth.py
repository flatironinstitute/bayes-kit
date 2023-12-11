import numpy as np

# code ported from arviz: https://python.arviz.org/en/stable/_modules/arviz/stats/stats.html#loo
def gen_pareto_estimate(x):
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

def gen_pareto_icdf(u, mu, sigma, kappa):
    return mu + sigma / kappa * ((1 - u)**(-kappa) - 1)

def pareto_smooth(weights_raw):
    weights = np.array(weights_raw)
    S = len(weights)
    M = int(np.minimum(0.2 * S, 3 * np.sqrt(S)))
    max_weight = np.max(weights)
    print(f"{max_weight = }")
    idxs = np.argpartition(-weights, M)[:M]
    print(f"{idxs = }")
    largest_weights = weights[idxs]
    print(f"{largest_weights = }")
    kappa_hat, sigma_hat = gen_pareto_estimate(largest_weights)
    print(f"{kappa_hat = }   {sigma_hat = }")
    ranks = np.argsort(np.argsort(largest_weights))
    print(f"{ranks = }")
    mu_hat = np.sort(weights)[-M]
    weights[idxs] = [np.minimum(max_weight, gen_pareto_icdf((rank + 0.5) / M, mu_hat, sigma_hat, kappa_hat)) for rank in ranks]
    return weights, kappa_hat

    

