"""Logging utilities for predictions."""

from pathlib import Path
from datetime import datetime


def log_predictions(batch_df, predictions, probabilities, batch_name, output_dir="reports/predictions"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = batch_df.copy()
    result_df["prediction"] = predictions

    if probabilities is not None:
        if len(probabilities.shape) == 1:
            result_df["prediction_probability"] = probabilities
        else:
            result_df["prediction_probability"] = probabilities[:, 1]

    result_df["batch_name"] = batch_name
    result_df["prediction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_path = output_dir / f"{Path(batch_name).stem}_predictions.csv"
    result_df.to_csv(output_path, index=False)

    return output_path
