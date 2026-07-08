"""Batch discovery and ordering utilities for incoming data."""

from pathlib import Path


def list_batch_files(metadata):
    incoming_dir = Path(metadata["incoming_data_dir"])
    file_format = metadata.get("file_format", "csv").lower()

    if not incoming_dir.exists():
        raise FileNotFoundError(f"Incoming data directory not found: {incoming_dir}")

    batch_files = list(incoming_dir.glob(f"*.{file_format}"))

    if not batch_files:
        raise FileNotFoundError(f"No .{file_format} batch files found in {incoming_dir}")

    return batch_files


def get_sorted_batches(metadata, pipeline_config):
    batch_files = list_batch_files(metadata)

    sorting_strategy = (
        pipeline_config.get("batch_settings", {}).get("batch_sorting", "filename")
    )

    if sorting_strategy == "filename":
        batch_files = sorted(batch_files, key=lambda x: x.name)
    elif sorting_strategy == "created_time":
        batch_files = sorted(batch_files, key=lambda x: x.stat().st_ctime)
    elif sorting_strategy == "modified_time":
        batch_files = sorted(batch_files, key=lambda x: x.stat().st_mtime)
    else:
        raise ValueError(f"Unsupported batch sorting strategy: {sorting_strategy}")

    return [str(path) for path in batch_files]
