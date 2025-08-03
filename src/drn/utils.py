from typing import Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch


def split_and_preprocess(
    features, target, num_features, cat_features, seed=42, num_standard=True
):
    # Before preprocessing split
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        features, target, random_state=seed, train_size=0.8, shuffle=True
    )

    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_train_raw, y_train, random_state=seed, train_size=0.75, shuffle=True
    )

    # Determine the full set of categories for each feature
    all_categories = {feature: set() for feature in cat_features}
    for feature in cat_features:
        all_categories[feature].update(features[feature].unique())

    # Convert each categorical feature to a categorical type with all possible categories
    for feature in cat_features:
        # Sort the categories to ensure consistent order
        sorted_categories = sorted(all_categories[feature])
        x_train_raw[feature] = pd.Categorical(
            x_train_raw[feature], categories=sorted_categories
        )
        x_val_raw[feature] = pd.Categorical(
            x_val_raw[feature], categories=sorted_categories
        )
        x_test_raw[feature] = pd.Categorical(
            x_test_raw[feature], categories=sorted_categories
        )
        features[feature] = pd.Categorical(
            features[feature], categories=sorted_categories
        )

    # One-hot Encoding
    features_one_hot = pd.get_dummies(features, columns=cat_features, drop_first=True)
    features_one_hot = features_one_hot.astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        features_one_hot, target, random_state=seed, train_size=0.8, shuffle=True
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, random_state=seed, train_size=0.75, shuffle=True
    )

    if num_standard:
        # Standarise the numeric features.
        ct = ColumnTransformer(
            [("standardize", StandardScaler(), num_features)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        x_train = ct.fit_transform(x_train)
        x_val = ct.transform(x_val)
        x_test = ct.transform(x_test)
    else:
        ct = None

    x_train = pd.DataFrame(
        x_train, index=x_train_raw.index, columns=features_one_hot.columns
    )
    x_val = pd.DataFrame(x_val, index=x_val_raw.index, columns=features_one_hot.columns)
    x_test = pd.DataFrame(
        x_test, index=x_test_raw.index, columns=features_one_hot.columns
    )

    return (
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
    )


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    seed: int = 42,
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split features and target into train, validation, and test sets based on fractions of the entire dataset.

    Args:
        features: DataFrame of predictors.
        target: Series of labels.
        seed: Random seed for reproducibility.
        train_size: Fraction of data for training.
        val_size: Fraction of data for validation.
            (test_size is computed as 1 - train_size - val_size)
    Returns:
        x_train_raw, x_val_raw, x_test_raw,
        y_train, y_val, y_test
    """
    # Compute test fraction
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError(
            f"train_size + val_size must be < 1. Got {train_size + val_size}"
        )

    # Split off test set
    x_train_val, x_test_raw, y_train_val, y_test = train_test_split(
        features, target, test_size=test_size, random_state=seed, shuffle=True
    )
    # Split train+val into train and val
    relative_val_size = val_size / (train_size + val_size)
    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        shuffle=True,
    )
    return x_train_raw, x_val_raw, x_test_raw, y_train, y_val, y_test


def replace_rare_categories(
    df: pd.DataFrame,
    threshold: int = 10,
    placeholder: str = "OTHER",
    cat_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Replace rare categories in specified categorical columns with a placeholder category.

    Parameters:
    - df: The input DataFrame.
    - threshold: Minimum number of occurrences for a category to be kept.
    - placeholder: Name to assign to rare categories.
    - cat_features: If specified, only apply to these columns.

    Raises:
    - ValueError: If the placeholder value already exists in any of the target columns.

    Returns:
    - pd.DataFrame: A new DataFrame with rare categories replaced.
    """
    df = df.copy()
    columns = (
        cat_features
        if cat_features is not None
        else df.select_dtypes(include=["object", "category"]).columns
    )

    # Check for placeholder conflicts
    for col in columns:
        if placeholder in df[col].unique():
            raise ValueError(
                f"The placeholder value '{placeholder}' already exists in column '{col}'."
            )

    for col in columns:
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < threshold].index
        df[col] = df[col].apply(lambda x: placeholder if x in rare_categories else x)
        if df[col].dtype.name != "category":
            df[col] = df[col].astype("category")
    return df


def generate_categories(
    x_train_raw: pd.DataFrame,
    x_val_raw: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_features: list[str],
) -> dict[str, list]:
    """
    Create a mapping of categorical features to their full category lists:
      - Initialize from training split
      - Detect new categories in val/test, print a warning, and extend
    Returns:
        all_categories: feature -> sorted list of categories
    """
    all_categories = {
        feature: sorted(x_train_raw[feature].dropna().unique())
        for feature in cat_features
    }
    for split_name, df in [("validation", x_val_raw), ("test", x_test)]:
        for feature in cat_features:
            seen = set(all_categories[feature])
            unique_vals = set(df[feature].dropna().unique())
            new_vals = unique_vals - seen
            if new_vals:
                print(
                    f"New categories for '{feature}' in {split_name} split: {new_vals}"
                )
                all_categories[feature] = sorted(
                    all_categories[feature] + list(new_vals)
                )
    return all_categories


def preprocess_data(
    x_train_raw: pd.DataFrame,
    x_val_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    num_standard: bool = True,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer, dict[str, list]
]:
    """
    Fit a ColumnTransformer on x_train_raw and transform raw splits.
    - Numeric features are optionally standardized.
    - Categorical features are one-hot encoded, using full categories detected from splits.

    Returns:
        x_train, x_val, x_test, fitted ColumnTransformer, all_categories mapping
    """
    # Prepare category mapping
    all_categories = generate_categories(
        x_train_raw, x_val_raw, x_test_raw, cat_features
    )

    # OneHotEncoder with fixed categories
    ohe = OneHotEncoder(
        categories=[all_categories[f] for f in cat_features],
        handle_unknown="error",
        sparse_output=False,
        drop="first",
    )

    # Build transformers list
    transformers = [
        ("num", StandardScaler() if num_standard else "passthrough", num_features),
        ("cat", ohe, cat_features),
    ]
    ct = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )

    # Fit & transform splits
    x_train_arr = ct.fit_transform(x_train_raw)
    x_val_arr = ct.transform(x_val_raw)
    x_test_arr = ct.transform(x_test_raw)

    # Build DataFrames with proper feature names
    feature_names = ct.get_feature_names_out()
    x_train = pd.DataFrame(x_train_arr, columns=feature_names, index=x_train_raw.index)
    x_val = pd.DataFrame(x_val_arr, columns=feature_names, index=x_val_raw.index)
    x_test = pd.DataFrame(x_test_arr, columns=feature_names, index=x_test_raw.index)

    return x_train, x_val, x_test, ct, all_categories


def _to_numpy(data):
    """Convert input data to numpy array with float32 precision."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float32)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values.astype(np.float32)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float32)
    else:
        return np.asarray(data, dtype=np.float32)
