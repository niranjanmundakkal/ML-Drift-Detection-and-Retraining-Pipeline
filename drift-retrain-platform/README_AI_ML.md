# AI/ML README

## Overview
This part of the project implements the machine learning workflow for a drift-aware churn classification pipeline. It trains a baseline model, scores incoming batches, detects drift, evaluates performance, and decides whether retraining should be triggered.

## Goals
- Train a baseline classifier on reference data
- Generate predictions for incoming batches
- Measure drift over time
- Evaluate model quality and trigger candidate retraining
- Promote better-performing models when appropriate

## Pipeline Flow
1. Load configuration and metadata
2. Load reference data
3. Preprocess and transform the reference set
4. Train a baseline model
5. Process incoming batches one by one
6. Generate predictions and probabilities
7. Run drift detection against the reference distribution
8. Evaluate the current model and possible retraining candidates
9. Promote a candidate model if it performs better

## Main AI/ML Components
### Model training
- `src/models/train.py`
  - trains the baseline model
  - trains a retraining candidate model

### Prediction
- `src/models/predict.py`
  - runs model inference on incoming batch features

### Evaluation
- `src/models/evaluate.py`
  - computes accuracy, precision, recall, F1, and ROC AUC

### Drift detection
- `src/drift/detector.py`
  - compares reference and batch feature distributions
  - produces a drift report per batch

### Retraining policy
- `src/retraining/policy.py`
  - decides whether drift is strong enough to trigger retraining

## Current Model Choice
The current implementation uses a scikit-learn `RandomForestClassifier` as the baseline and retraining model.

## Main Entry Point
Run the full workflow with:
```bash
python src/main.py
```

## What Happens at Runtime
- The reference dataset is used to train the initial model
- Each batch is transformed with the fitted preprocessor
- Predictions are logged to `reports/predictions/`
- Drift is logged to `reports/drift_reports/`
- Metrics are logged to `reports/metrics/`

## Evaluation Metrics
The model evaluation step currently reports:
- accuracy
- precision
- recall
- F1 score
- ROC AUC (when available)

## Retraining Logic
Retraining is triggered when the drift report indicates significant change in the batch distribution. A candidate model is then evaluated and optionally promoted if it improves the selected primary metric.

## Notes
This AI/ML layer is intentionally modular so it can be extended with:
- alternative classifiers
- more advanced drift detectors
- deeper retraining strategies
- deployment-ready model versioning

## Suggested Next Improvements
- add a validation split before retraining
- persist the best model to disk
- introduce confidence-based thresholds
- add explainability outputs for predictions and drift
