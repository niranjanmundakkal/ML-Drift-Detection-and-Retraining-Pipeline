"""Build metadata for datasets and models from schema and config files."""


def build_metadata(schema, pipeline_config, model_config):
    metadata = {
        "dataset_name": schema.get("dataset_name"),
        "task_type": schema.get("task_type"),
        "target_column": schema.get("target_column"),
        "id_column": schema.get("id_column"),
        "feature_columns": schema.get("feature_columns", []),
        "numerical_columns": schema.get("numerical_columns", []),
        "categorical_columns": schema.get("categorical_columns", []),
        "datetime_columns": schema.get("datetime_columns", []),
        "drop_columns": schema.get("drop_columns", []),
        "label_available_in_batches": schema.get("label_available_in_batches", True),
        "reference_data_path": pipeline_config.get("data_settings", {}).get("reference_data_path"),
        "incoming_data_dir": pipeline_config.get("data_settings", {}).get("incoming_data_dir"),
        "processed_data_dir": pipeline_config.get("data_settings", {}).get("processed_data_dir"),
        "file_format": pipeline_config.get("data_settings", {}).get("file_format", "csv"),
        "train_test_split": pipeline_config.get("split_settings", {}).get("train_test_split", 0.2),
        "random_state": pipeline_config.get("split_settings", {}).get("random_state", 42),
        "stratify": pipeline_config.get("split_settings", {}).get("stratify", True),
        "drift_threshold": model_config.get("drift_detection", {}).get("threshold", 0.2),
        "evaluation_metrics": model_config.get("evaluation", {}).get("metrics", []),
        "primary_metric": model_config.get("evaluation", {}).get("primary_metric"),
        "model_type": model_config.get("model", {}).get("model_type"),
        "model_params": model_config.get("model", {}).get("model_params", {}),
    }
    return metadata
