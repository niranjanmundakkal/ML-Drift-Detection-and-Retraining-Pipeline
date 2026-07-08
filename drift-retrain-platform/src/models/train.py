"""Training utilities for the orchestration pipeline."""

from sklearn.ensemble import RandomForestClassifier


def train_baseline_model(X_ref, y_ref, metadata=None, model_config=None):
    model_type = (model_config or {}).get("model", {}).get("model_type", "random_forest")
    params = (model_config or {}).get("model", {}).get("model_params", {})

    if model_type == "random_forest":
        model = RandomForestClassifier(**params)
    else:
        model = RandomForestClassifier(**params)

    model.fit(X_ref, y_ref)
    return model


def retrain_model(X_ref, y_ref, X_batch, y_batch, metadata=None, model_config=None):
    return train_baseline_model(X_ref, y_ref, metadata=metadata, model_config=model_config)
