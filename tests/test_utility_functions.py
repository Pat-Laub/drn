import numpy as np
import pandas as pd
import pytest
from drn import *
from synthetic_dataset import generate_synthetic_data


# Sample test data and scenarios we discussed
@pytest.mark.parametrize(
    "L_Raw, data_train, M, expected",
    [
        # Test Case 1: Small dataset with clear cutpoints
        (
            [0, 3, 6, 9],
            np.array([0.5, 2.5, 4.5, 6.5, 8.5]),
            2,
            [0, 3, 9],
        ),  # As derived from our correct implementation
        # Test Case 2: Problematic case for the old algorithm where the final bucket might have less than M observations
        (
            [0, 3, 7, 9],
            np.array([0.5, 2.5, 4.5, 6.5, 8.5]),
            2,
            [0, 3, 9],
        ),  # Adjusted expectation to fit the correct algorithm output
    ],
)
def test_merge_cutpoints(L_Raw, data_train, M, expected):
    assert merge_cutpoints(L_Raw, data_train, M) == expected


def test_split_preprocess():

    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_data()

    # Check we avoid this 'iloc' warning when training on pandas data types
    X_df = pd.DataFrame(
        X_train.detach().cpu().numpy(),
        columns=[f"X_{i}" for i in range(X_train.shape[1])],
    )
    y_ser = pd.Series(Y_train.detach().cpu().numpy(), name="Y")

    # Check that pandas objects from split_and_preprocess
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        x_train_raw,
        x_val_raw,
        x_test_raw,
        num_features,
        cat_features,
        all_categories,
        ct,
    ) = split_and_preprocess(
        X_df, y_ser, ["X_0", "X_1", "X_2", "X_3"], [], seed=42, num_standard=True
    )
