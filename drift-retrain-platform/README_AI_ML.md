# AI/ML README

# Overview

This module implements the Machine Learning workflow for a **dynamic drift-aware retraining pipeline**. The pipeline continuously monitors incoming batches for data drift, automatically evaluates whether retraining is required, trains multiple candidate models, selects the best-performing model, and promotes it if it outperforms the existing production model.

The AI/ML layer is schema-driven, making it reusable across multiple machine learning use cases without changing the core pipeline.

---

# Supported Use Cases

The pipeline currently supports two configurable ML applications:

- Customer Churn Prediction
- Credit Risk Assessment

Switching between use cases only requires changing the active configuration.

---

# Objectives

- Train an initial baseline model
- Score incoming data batches
- Monitor feature drift
- Trigger retraining automatically when required
- Train multiple candidate models
- Automatically select the best-performing model
- Promote improved models into production
- Generate prediction, evaluation, and drift reports

---

# End-to-End Pipeline

```
Reference Dataset
        │
        ▼
Dynamic Preprocessing
        │
        ▼
Train Baseline Model
        │
        ▼
Production Model
        │
====================================================
Incoming Batch
        │
        ▼
Dynamic Preprocessing
        │
        ▼
Prediction
        │
        ▼
Drift Detection
        │
        ▼
Retraining Policy
        │
        ▼
If Drift Threshold Exceeded
        │
        ▼
Train Candidate Models
        │
        ▼
Automatic Model Selection
        │
        ▼
Candidate Promotion
```

---

# AI/ML Components

## Dynamic Preprocessing

**src/preprocessing/preprocessor.py**

- Schema-driven preprocessing
- Missing value handling
- Numerical scaling
- One-hot encoding
- Dynamic feature generation

---

## Model Training

**src/models/train.py**

Responsible for:

- Baseline model training
- Candidate model retraining
- Automatic model comparison
- Best model selection using validation F1-score

Current candidate models:

- Random Forest
- Logistic Regression
- Decision Tree

The pipeline automatically selects the highest-performing model.

---

## Prediction

**src/models/predict.py**

- Batch inference
- Probability prediction
- Feature alignment
- Prediction logging

Outputs are stored in:

```
reports/predictions/
```

---

## Drift Detection

**src/drift/**

Supports configurable drift detection for both numerical and categorical features.

Numerical methods:

- Population Stability Index (PSI)
- Kolmogorov–Smirnov (KS) Test

Categorical methods:

- Categorical PSI
- Distribution comparison

Each feature is assigned:

- Drift score
- Severity level
- Drift status

Reports are saved to:

```
reports/drift_reports/
```

---

## Evaluation

**src/models/evaluate.py**

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The primary metric is configurable through:

```
model_config.json
```

---

## Retraining Policy

**src/retraining/policy.py**

The retraining policy automatically determines whether retraining should occur based on:

- Number of drifted features
- Drift percentage
- Configurable drift threshold

Possible outcomes:

- Continue Monitoring
- Trigger Retraining

---

## Model Promotion

After retraining:

1. Candidate models are evaluated.
2. The best-performing model is selected.
3. The candidate is compared with the production model.
4. Better models are promoted automatically.

---

# Runtime Workflow

Executing

```bash
python src/main.py
```

performs the following:

1. Load configuration
2. Load schema
3. Load reference dataset
4. Dynamic preprocessing
5. Train baseline model
6. Process incoming batches
7. Generate predictions
8. Detect feature drift
9. Decide retraining
10. Train candidate models
11. Select best-performing model
12. Promote candidate model if improved
13. Save reports

---

# Output Reports

Prediction reports

```
reports/predictions/
```

Drift reports

```
reports/drift_reports/
```

Evaluation metrics

```
reports/metrics/
```

---

# Configuration Driven

The AI/ML pipeline is completely configuration-driven.

Configuration files define:

- active use case
- preprocessing strategy
- drift detection method
- drift thresholds
- evaluation metrics
- retraining policy
- model selection strategy

No code changes are required when switching supported use cases.

---

# Current Features

✔ Dynamic preprocessing

✔ Automatic model selection

✔ Multiple candidate models

✔ Configurable drift detection

✔ Automatic retraining

✔ Candidate model promotion

✔ Customer Churn pipeline

✔ Credit Risk pipeline

✔ Synthetic dataset generation

✔ Prediction logging

✔ Drift reporting

✔ Schema-driven architecture

---

# Future Enhancements

- Model versioning
- MLflow experiment tracking
- SHAP explainability
- Docker deployment
- REST API serving
- Real-time data streaming
- Monitoring dashboard
- Kubernetes deployment

---

# AI/ML Highlights

- Generic schema-driven architecture
- Supports multiple business use cases
- Automatic drift monitoring
- Automatic retraining decisions
- Automatic best-model selection
- Modular and easily extensible design
