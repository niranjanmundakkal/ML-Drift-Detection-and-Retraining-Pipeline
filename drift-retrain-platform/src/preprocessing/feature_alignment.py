"""Feature alignment utilities for aligned model input."""

import pandas as pd


def align_features(X_batch, training_feature_names):
    if not isinstance(X_batch, pd.DataFrame):
        raise ValueError("X_batch must be a pandas DataFrame for feature alignment.")

    X_batch = X_batch.copy()

    for col in training_feature_names:
        if col not in X_batch.columns:
            X_batch[col] = 0

    extra_columns = [col for col in X_batch.columns if col not in training_feature_names]
    if extra_columns:
        X_batch = X_batch.drop(columns=extra_columns)

    X_batch = X_batch[training_feature_names]

    return X_batch
