"""Logging utilities for drift reports."""

from pathlib import Path
from datetime import datetime


def log_drift_report(drift_report_df, batch_name, output_dir="reports/drift_reports"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drift_report_df = drift_report_df.copy()
    drift_report_df["batch_name"] = batch_name
    drift_report_df["drift_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_path = output_dir / f"{Path(batch_name).stem}_drift_report.csv"
    drift_report_df.to_csv(output_path, index=False)

    return output_path
