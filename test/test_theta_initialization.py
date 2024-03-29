from typing import Any, List
from unittest.mock import Mock

import numpy as np

from bayes_kit.hmc import HMCDiag
from bayes_kit.mala import MALA
from bayes_kit.metropolis import Metropolis, MetropolisHastings
from bayes_kit.typing import VectorType


def assert_not_empty_array(a: VectorType) -> None:
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(a, np.array([]))


def make_models(init: VectorType, dims: int = 1) -> List[Any]:
    # For the purposes of this (and future) tests, we only care about the model's dimensions.
    mock_model: Any = Mock()
    mock_model.dims = Mock(return_value=dims)
    mock_model.log_density_gradient = Mock(return_value=(0.5, (0,)))
    mock_model.log_density_hessian = Mock(return_value=(0, 5, (0,), (0,)))

    # TODO: Add ensemble/stretcher once the class is available
    hmc = HMCDiag(mock_model, stepsize=0.25, steps=10, init=init)
    mala = MALA(mock_model, epsilon=0.5, init=init)
    metro = Metropolis(mock_model, lambda x: 1, init=init)
    mh = MetropolisHastings(mock_model, lambda x: 1, lambda x, y: 1, init=init)
    # TemperedLikelihoodSMC omitted since it does not have explicit-value initialization of thetas

    return [hmc, mala, metro, mh]


def test_init_with_empty_numpy_array() -> None:
    # an "empty" initialization should be treated as no initialization
    # (i.e. not honored)
    init = np.array([])
    models = make_models(init)
    for model in models:
        assert_not_empty_array(model._theta)


def test_init_with_one_element_numpy_array() -> None:
    init = np.array([3])
    models = make_models(init)
    for model in models:
        np.testing.assert_array_equal(model._theta, init)


def test_init_with_multi_element_numpy_array() -> None:
    init = np.array([3, 3, 3])
    models = make_models(init, dims=3)
    for model in models:
        np.testing.assert_array_equal(model._theta, init)
