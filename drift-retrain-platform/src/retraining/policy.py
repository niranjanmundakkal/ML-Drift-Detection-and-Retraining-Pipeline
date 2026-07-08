"""Retraining policy helpers."""

import pandas as pd


def should_retrain(drift_report_df, model_config=None):
    if drift_report_df is None or drift_report_df.empty:
        return False

    threshold = (model_config or {}).get("drift_detection", {}).get("threshold", 0.05)
    drift_hits = drift_report_df.get("drift_detected", pd.Series([False] * len(drift_report_df)))
    return bool(drift_hits.sum() > 0 and drift_hits.mean() >= threshold)
