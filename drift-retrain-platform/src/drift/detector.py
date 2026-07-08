"""Simple drift detection utilities."""

import numpy as np
import pandas as pd
from scipy import stats


def compute_drift_report(X_reference, X_batch, metadata=None, model_config=None):
    if X_reference is None or X_batch is None:
        return pd.DataFrame(columns=["feature", "statistic", "p_value", "drift_detected"])

    reference_array = np.asarray(X_reference)
    batch_array = np.asarray(X_batch)

    if reference_array.ndim == 1:
        reference_array = reference_array.reshape(-1, 1)
    if batch_array.ndim == 1:
        batch_array = batch_array.reshape(-1, 1)

    if reference_array.shape[1] != batch_array.shape[1]:
        n_features = min(reference_array.shape[1], batch_array.shape[1])
        reference_array = reference_array[:, :n_features]
        batch_array = batch_array[:, :n_features]

    rows = []
    for idx in range(reference_array.shape[1]):
        ref_col = reference_array[:, idx]
        batch_col = batch_array[:, idx]
        statistic, p_value = stats.ks_2samp(ref_col, batch_col)
        rows.append({
            "feature": idx,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < (metadata or {}).get("drift_threshold", 0.05)),
        })

    return pd.DataFrame(rows)
