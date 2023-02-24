from test.models.std_normal import StdNormal
from bayes_kit.ensemble import AffineInvariantWalker
import numpy as np
import pytest as pt

def run_iterator_test(sampler, model) -> None:
    with np.testing.assert_no_warnings():
        i = 0
        for m in sampler:
            # make sure we can take 10 draws
            if i > 10:
                break
            i += 1

def run_sampling_test(sampler, model) -> None:
    run_iterator_test(sampler, model)
    D = sampler._dim
    K = sampler._num_walkers
    M = 10000
    draws = np.ndarray(shape=(M, K, D))
    for m in range(M):
        draws[m, 0:K, 0:D] = sampler.sample()
    mean = np.mean(draws)
    var = np.var(draws, ddof=1)
    # sampler super inefficient with these settings, so need wide tolerance
    # longer tests with M = 100_000 will converge much better but take several seconds
    np.testing.assert_allclose(mean, model.posterior_mean(), atol=0.2)
    np.testing.assert_allclose(var, model.posterior_variance(), atol=0.2)

def test_aiw_exceptions() -> None:
    model = StdNormal()
    # illegal value: a
    with pt.raises(ValueError):
        sampler = AffineInvariantWalker(model, a = -1)
    with pt.raises(ValueError):
        sampler = AffineInvariantWalker(model, a = 0)
    with pt.raises(ValueError):
        sampler = AffineInvariantWalker(model, a = 1)

    # illegal value: num_walkers
    with pt.raises(ValueError):
        sampler = AffineInvariantWalker(model, num_walkers = -1)
    with pt.raises(ValueError):
        sampler = AffineInvariantWalker(model, num_walkers = 0)
    with pt.raises(ValueError):
        sampler = AffineInvariantWalker(model, num_walkers = 1)

    # illegal value: init
    with pt.raises(ValueError):
        sampler = AffineInvariantWalker(model, init = np.asarray([1.2, 2, 3]))

    # illegal type: a
    with pt.raises(TypeError):
        sampler = AffineInvariantWalker(model, a = (1, 2, 3))
        
    # illegal type: seed
    with pt.raises(TypeError):
        sampler = AffineInvariantWalker(model, seed=1.234)

    # illegal type: num_walkers
    with pt.raises(TypeError):
        sampler = AffineInvariantWalker(model, num_walkers = 2.39)

    # illegal type: init
    with pt.raises(TypeError):
        sampler = AffineInvariantWalker(model, init = [1.2, 3.9])

    # illegal type: model
    model_dummy = "abc"
    with pt.raises(AttributeError):
        sampler = AffineInvariantWalker(model_dummy)

        
def test_aiw_std_normal() -> None:
    model = StdNormal()
    # default config
    sampler = AffineInvariantWalker(model)
    run_sampling_test(sampler, model)

    # specifying bounds a
    sampler = AffineInvariantWalker(model, a = 2)
    run_sampling_test(sampler, model)

    # specifying num_walkers
    sampler = AffineInvariantWalker(model, num_walkers=6)
    run_sampling_test(sampler, model)

    # specifying init
    nw = 4
    sampler = AffineInvariantWalker(model, num_walkers=nw, init=np.random.normal(size=(nw, model.dims())))
    run_sampling_test(sampler, model)

    # specifying seed as int
    sampler = AffineInvariantWalker(model, seed=1234)
    run_sampling_test(sampler, model)

    # specifying seed as np.random.BitGenerator
    sampler = AffineInvariantWalker(model, seed=np.random.MT19937())
    run_sampling_test(sampler, model)

    # specifying seed as np.random.Generator
    sampler = AffineInvariantWalker(model, seed=np.random.default_rng())
    run_sampling_test(sampler, model)
    
