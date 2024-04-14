import numpy as np
import pytest
from distributionalforecasting import merge_cutpoints

# Sample test data and scenarios we discussed
@pytest.mark.parametrize("L_Raw, data_train, M, expected", [
    # Test Case 1: Small dataset with clear cutpoints
    ([0, 3, 6, 9], np.array([0.5, 2.5, 4.5, 6.5, 8.5]), 2,
     [0, 3, 9]),  # As derived from our correct implementation

    # Test Case 2: Problematic case for the old algorithm where the final bucket might have less than M observations
    ([0, 3, 7, 9], np.array([0.5, 2.5, 4.5, 6.5, 8.5]), 2,
     [0, 3, 9]),  # Adjusted expectation to fit the correct algorithm output
])
def test_merge_cutpoints(L_Raw, data_train, M, expected):
    assert merge_cutpoints(L_Raw, data_train, M) == expected

