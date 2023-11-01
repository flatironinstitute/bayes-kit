import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _set_test_seed():
    # Set the random seed before each test. Seed chosen by keyboard-mashing.
    np.random.seed(246784289)
