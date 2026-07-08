"""Validation helpers for reference data and incoming batches."""


def _check_duplicate_columns(df):
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate columns found: {duplicated}")


def _check_feature_group_consistency(schema):
    feature_columns = set(schema.get("feature_columns", []))
    numerical_columns = set(schema.get("numerical_columns", []))
    categorical_columns = set(schema.get("categorical_columns", []))
    datetime_columns = set(schema.get("datetime_columns", []))
    drop_columns = set(schema.get("drop_columns", []))

    if not numerical_columns.issubset(feature_columns):
        missing = numerical_columns - feature_columns
        raise ValueError(f"Numerical columns not present in feature_columns: {missing}")

    if not categorical_columns.issubset(feature_columns):
        missing = categorical_columns - feature_columns
        raise ValueError(f"Categorical columns not present in feature_columns: {missing}")

    if not datetime_columns.issubset(feature_columns):
        missing = datetime_columns - feature_columns
        raise ValueError(f"Datetime columns not present in feature_columns: {missing}")

    overlap = drop_columns.intersection(feature_columns)
    if overlap:
        raise ValueError(f"Columns cannot be in both drop_columns and feature_columns: {overlap}")


def validate_reference_schema(df, schema):
    _check_duplicate_columns(df)
    _check_feature_group_consistency(schema)

    feature_columns = set(schema.get("feature_columns", []))
    target_column = schema.get("target_column")
    drop_columns = set(schema.get("drop_columns", []))

    required_columns = feature_columns.union({target_column}) - drop_columns
    df_columns = set(df.columns)

    missing_columns = required_columns - df_columns
    if missing_columns:
        raise ValueError(f"Reference dataset is missing required columns: {missing_columns}")

    return True


def validate_batch_schema(df, schema):
    _check_duplicate_columns(df)
    _check_feature_group_consistency(schema)

    feature_columns = set(schema.get("feature_columns", []))
    target_column = schema.get("target_column")
    drop_columns = set(schema.get("drop_columns", []))
    label_available_in_batches = schema.get("label_available_in_batches", True)

    required_columns = feature_columns - drop_columns
    df_columns = set(df.columns)

    missing_columns = required_columns - df_columns
    if missing_columns:
        raise ValueError(f"Incoming batch is missing required feature columns: {missing_columns}")

    if label_available_in_batches and target_column not in df_columns:
        raise ValueError(
            f"Incoming batch is expected to contain target column '{target_column}', but it is missing."
        )

    return True
