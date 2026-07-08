"""Config-driven dataset loading utilities."""

from pathlib import Path
import pandas as pd


def _read_file(path, file_format="csv"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if file_format.lower() == "csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {file_format}")


def _apply_drop_columns(df, schema):
    drop_columns = schema.get("drop_columns", [])
    existing_drop_columns = [col for col in drop_columns if col in df.columns]

    if existing_drop_columns:
        df = df.drop(columns=existing_drop_columns)

    return df


def load_reference_data(metadata, schema):
    reference_path = metadata["reference_data_path"]
    file_format = metadata.get("file_format", "csv")

    df = _read_file(reference_path, file_format)
    df = _apply_drop_columns(df, schema)

    return df


def load_batch_data(batch_path, metadata, schema):
    file_format = metadata.get("file_format", "csv")

    df = _read_file(batch_path, file_format)
    df = _apply_drop_columns(df, schema)

    return df
