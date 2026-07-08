"""Evaluation utilities for the orchestration pipeline."""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(model, X, y, metadata=None):
    predictions = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
    }

    try:
        probabilities = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y, probabilities)
    except Exception:
        metrics["roc_auc"] = None

    return metrics
