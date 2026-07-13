"""Prediction utilities for the orchestration pipeline."""


def predict_with_model(model, X_batch):
    predictions = model.predict(X_batch)
    probabilities = model.predict_proba(X_batch)[:, 1]
    return predictions, probabilities
