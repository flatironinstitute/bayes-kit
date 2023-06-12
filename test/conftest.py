import pytest
import numpy as np
import scipy.stats as sst
from test.models import Binomial, SkewNormal, StdNormal
from bayes_kit.algorithms import HMCDiag, MALA, MetropolisHastings, Metropolis


@pytest.fixture(
    params=[Binomial(alpha=2, beta=3, x=5, N=15), SkewNormal(a=4), StdNormal()],
    ids=["Binomial", "SkewNormal", "StdNormal"],
)
def grad_model(request):
    return request.param


@pytest.fixture(params=[Binomial(alpha=2, beta=3, x=5, N=15)], ids=["Binomial"])
def log_prior_likelihood_model(request):
    return request.param


_hmc_init_kwargs = {"stepsize": 0.01, "steps": 100, "seed": 461145}
_mala_init_kwargs = {"epsilon": 0.01, "seed": 361484}
_mh_init_kwargs = {
    "proposal_fn": lambda theta, rng: np.array(
        [sst.skewnorm.rvs(4, loc=theta, random_state=rng)]
    ),
    "transition_lp_fn": lambda observation, given: sst.skewnorm.logpdf(
        observation, 4, loc=given
    )[0],
    "seed": 1298815,
}
_met_init_kwargs = {
    "proposal_fn": lambda theta, rng: np.array(
        [sst.skewnorm.rvs(4, loc=theta, random_state=rng)]
    ),
    "seed": 189189,
}


@pytest.fixture(
    params=[
        (HMCDiag, _hmc_init_kwargs),
        (MALA, _mala_init_kwargs),
        (MetropolisHastings, _mh_init_kwargs),
        (Metropolis, _met_init_kwargs),
    ],
    ids=["HMCDiag", "MALA", "MetropolisHastings", "Metropolis"],
)
def algorithm_cls_and_kwargs(request):
    return request.param


@pytest.fixture(
    params=[
        (HMCDiag, HMCDiag.Params(**_hmc_init_kwargs), {}),
        (MALA, MALA.Params(**_mala_init_kwargs), {}),
        (MetropolisHastings, MetropolisHastings.Params(seed=_mh_init_kwargs["seed"]), _mh_init_kwargs),
        (Metropolis, Metropolis.Params(seed=_met_init_kwargs["seed"]), _met_init_kwargs),
    ],
    ids=["HMCDiag", "MALA", "MetropolisHastings", "Metropolis"],
)
def algorithm_cls_and_params_and_kwargs(request):
    return request.param
