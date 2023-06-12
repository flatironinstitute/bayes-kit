
def test_algorithm_state_repeatable(algorithm_cls_and_kwargs, grad_model):
    cls, kwargs = algorithm_cls_and_kwargs
    algo1 = cls(model=grad_model, **kwargs)
    algo2 = cls(model=grad_model, **kwargs)

    # From initialization, they should have the same state
    init_state1 = algo1.get_state()
    assert init_state1 == algo2.get_state()

    # From after a bunch of steps, they should have different states
    for _ in range(10):
        step_result1, _ = algo1.step()
    assert algo1.get_state() != algo2.get_state()

    # After the same number of steps by algo2, they should match again
    for _ in range(10):
        step_result2, _ = algo2.step()
    assert algo1.get_state() == algo2.get_state()

    # Resetting algo1 back to initial state should make them not match
    algo1.set_state(init_state1)
    assert algo1.get_state() != algo2.get_state()

    # Running algo1 a second time from that initial state should make them match again
    for _ in range(10):
        step_result1, _ = algo1.step()
    assert algo1.get_state() == algo2.get_state()
