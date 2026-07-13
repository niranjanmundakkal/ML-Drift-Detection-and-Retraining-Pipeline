# Drift Retraining Platform

A production-style churn classification pipeline that automatically monitors incoming data, detects distribution drift, and retrains candidate models when needed.

## Why this project matters

This project demonstrates how production ML systems can automatically monitor incoming data, detect distribution shifts, retrain models, and safely promote better candidates without manual intervention.

That is essential for real-world models because data distributions change over time, and stale models can lose predictive quality quickly.

## Quick start

Visitors can run the full pipeline locally with the provided sample data generator and orchestration script.

```bash
git clone <repo-url>
cd drift-retrain-platform
python -m pip install -r requirements.txt
python scripts/generate_synthetic_customer_churn.py
python src/main.py
```

The pipeline will generate synthetic reference and incoming batch data, train the baseline model, score each batch, detect drift, and log outputs in `reports/`.

## Run with your own data

You can run the pipeline with your own CSV files instead of the generated sample data.

### What to provide

- `data/reference/reference.csv` : the reference dataset used to train the baseline model
- `data/incoming/*.csv` : one or more incoming batch files to score and monitor

### Required schema

Your data must include the target column and all feature columns defined in:

- `config/use_cases/customer_churn/schema.json`

For this project, the expected columns are:

- `customer_id`
- `age`
- `gender`
- `region`
- `senior_citizen`
- `partner`
- `dependents`
- `tenure`
- `contract`
- `internet_service`
- `tech_support`
- `payment_method`
- `monthly_charges`
- `total_charges`
- `avg_monthly_usage_gb`
- `support_calls_last_6m`
- `payment_delay_days`
- `churn`

### Useful tips

- Keep `customer_id` unique for each row.
- Use the same categorical categories across reference and batch files where possible.
- Missing values are handled automatically by the preprocessor, but consistent column names are required.
- If batch labels are not present, set `label_available_in_batches` to `false` in `config/use_cases/customer_churn/pipeline_config.json`.

### Run the pipeline

```bash
python src/main.py
```

### Custom schema or use case

If your dataset has different feature names, update:

- `config/use_cases/customer_churn/schema.json`
- `config/use_cases/customer_churn/pipeline_config.json`

This makes the pipeline friendly for new data and reuse in different churn or classification scenarios.

## Architecture

![Pipeline Architecture](docs/architecture_diagram.svg)

### High-level flow

1. Load reference and batch data.
2. Validate schema and prepare metadata.
3. Fit a preprocessing pipeline with imputation, encoding, and scaling.
4. Train a baseline model using automatic candidate selection.
5. Score incoming batches and log predictions.
6. Run drift detection using KS tests across transformed features.
7. If drift exceeds policy thresholds, retrain candidate models.
8. Promote a challenger only if it outperforms the current champion.

## Repository structure

```
src/
  core/             # config loading, metadata, schema validation
  ingestion/        # dataset loading and batch discovery
  logging_utils/    # prediction, drift, and metric logging
  models/           # training, inference, evaluation
  preprocessing/    # dynamic preprocessing and feature alignment
  drift/            # distribution drift detection
  retraining/       # retraining policy and candidate promotion
  training/         # training utilities and helpers
  main.py           # orchestration entry point
config/
  active_config.json
  use_cases/
    customer_churn/
      schema.json
      pipeline_config.json
      model_config.json
reports/
  drift_reports/
  metrics/
  predictions/
scripts/
  generate_synthetic_customer_churn.py
```

## Installation

```bash
git clone <repo-url>
cd drift-retrain-platform
python -m pip install -r requirements.txt
```

> If you prefer an isolated environment, create a virtual environment first:
>
> ```bash
> python -m venv .venv
> .venv\Scripts\activate
> python -m pip install -r requirements.txt
> ```

## Generate synthetic data

The synthetic data generator creates a reference dataset and several incoming batches with controlled drift.

```bash
python scripts/generate_synthetic_customer_churn.py
```

This produces:

- `data/reference/reference.csv`
- `data/incoming/batch_001.csv`
- `data/incoming/batch_002.csv`
- `data/incoming/batch_003.csv`
- `data/incoming/batch_004.csv`

## Run the pipeline

```bash
python src/main.py
```

The pipeline will:

- fit preprocessing on reference data
- train a baseline model with automatic selection across candidate classifiers
- score each incoming batch
- compute drift reports
- log metrics and predictions
- retrain and promote a challenger when drift is significant

## Configuration

The active use case is selected in `config/active_config.json`:

```json
{
  "active_use_case": "customer_churn"
}
```

### Pipeline settings: `config/use_cases/customer_churn/pipeline_config.json`

- `reference_data_path`: path to the reference dataset.
- `incoming_data_dir`: directory containing batch CSV files.
- `processed_data_dir`: optional processed output directory.
- `missing_value_strategy`: imputation strategy for numerical and categorical features.
- `encoding.categorical`: categorical encoding method.
- `scaling.numerical`: numerical feature scaling strategy.
- `train_test_split`: validation split ratio used during model selection.
- `label_available_in_batches`: whether incoming batches include target labels.

### Model settings: `config/use_cases/customer_churn/model_config.json`

- `model.model_type`: currently `automatic_selection`.
- `evaluation.primary_metric`: primary metric for candidate promotion.
- `evaluation.metrics`: metrics reported for baseline and candidate models.
- `drift_detection.method`: uses `ks_test` in this implementation.
- `drift_detection.threshold`: p-value threshold for KS drift detection.
- `retraining_policy.trigger_strategy`: drift-based retraining policy.
- `retraining_policy.minimum_improvement`: minimum improvement required to promote a challenger.

## Preprocessing pipeline

The pipeline includes:

- missing value imputation for numerical and categorical features
- categorical encoding with one-hot encoding
- numerical scaling with standard scaling
- feature alignment to preserve training feature order for incoming batches
- schema validation against `config/use_cases/customer_churn/schema.json`

## Drift detection

Drift is detected using:

- **Kolmogorov-Smirnov Test (KS Test)**: compares the distribution of each transformed feature between the reference dataset and an incoming batch.
- **Drift severity labels**: the implementation categorizes KS statistics into Low, Medium, or High.

A feature is marked as drifted when the KS p-value is below the configured threshold.

## Champion-Challenger model promotion

The current production model is the **Champion**.

When drift triggers retraining:

1. A candidate model is trained on the combined reference and batch data.
2. The candidate becomes the **Challenger**.
3. The challenger is evaluated on the selected primary metric.
4. The challenger is promoted only if it outperforms the current champion.

This ensures the model is improved safely rather than replaced blindly.

## Synthetic drift generation

The synthetic generator creates realistic drift in controlled stages by changing:

- customer senior citizen share
- contract type distribution
- tenure distribution
- monthly charges and total charges
- payment delay behavior
- missing value frequency

This helps demonstrate the pipeline's ability to detect drift and trigger retraining.

## Output artifacts

The pipeline writes:

- `reports/predictions/`: batch-level prediction exports
- `reports/drift_reports/`: feature-level drift analysis for each batch
- `reports/metrics/`: evaluation metrics for baseline and candidate models

## Sample outputs

```text
Batch: batch_002.csv
Drifted features: 12/25 (48.00%)
Retraining Triggered
Candidate model promoted for batch_002
```

## Contributions

### AI/ML

- model training and automatic candidate selection
- evaluation of accuracy, precision, recall, F1, and ROC AUC
- drift detection via KS test
- champion-challenger retraining strategy
- pipeline orchestration in `src/main.py`

### Data Engineering

- data ingestion and batch orchestration
- schema validation and preprocessing pipeline
- synthetic churn data generation
- feature alignment and dynamic preprocessing
- logging predictions, drift reports, and metrics
- config-driven pipeline behavior

## Future improvements

1. MLflow Model Registry
2. FastAPI model serving
3. Prefect/Airflow orchestration
4. Evidently AI dashboard
5. Streamlit monitoring dashboard
6. Docker packaging
7. GitHub Actions CI/CD
8. SHAP explainability
9. Kubernetes deployment
10. Feature store integration

## Related documents

- `README_AI_ML.md` – deep dive on the AI/ML workflow
- `README_DATA_ENGINEERING.md` – data engineering and ingestion details
