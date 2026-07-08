# Data Engineering README

## Overview
This module handles the data layer of the drift-retrain pipeline for synthetic customer churn data. It is responsible for loading configuration, validating schemas, reading reference and batch datasets, preprocessing features, and writing output artifacts for downstream ML steps.

## Goals
- Provide a config-driven data pipeline for reference and incoming batch data
- Validate input data before it enters the ML workflow
- Standardize preprocessing for tabular features
- Support synthetic data generation for drift experiments

## Project Structure
- `config/` – use-case configuration files for schema, pipeline settings, and model settings
- `data/` – reference data and incoming batch files
- `src/core/` – config and metadata utilities
- `src/ingestion/` – dataset loading and batch discovery
- `src/preprocessing/` – feature preprocessing and alignment
- `src/logging_utils/` – prediction, drift, and metrics logging
- `scripts/` – synthetic data generation utilities

## Core Components
### 1. Configuration layer
- `config/use_cases/customer_churn/schema.json` defines the expected data contract
- `config/use_cases/customer_churn/pipeline_config.json` defines input/output paths and preprocessing behavior
- `config/active_config.json` selects the active use case

### 2. Ingestion layer
- `src/ingestion/dataset_loader.py` loads reference and batch CSVs
- `src/ingestion/batch_manager.py` finds and orders incoming batch files

### 3. Validation layer
- `src/core/schema_validator.py` checks for:
  - required feature columns
  - target column presence in batches when expected
  - duplicate columns
  - consistency between feature, numerical, categorical, and datetime columns

### 4. Preprocessing layer
- `src/preprocessing/preprocessor.py` performs:
  - feature selection
  - imputation
  - one-hot encoding
  - scaling
- `src/preprocessing/feature_alignment.py` aligns batch features to the training feature order

### 5. Logging layer
- `src/logging_utils/prediction_logger.py` stores predictions
- `src/logging_utils/drift_logger.py` stores drift reports
- `src/logging_utils/metrics_logger.py` stores evaluation metrics

## Data Expectations
The pipeline expects:
- a reference dataset at `data/reference/reference.csv`
- incoming batch files inside `data/incoming/`

The current synthetic customer churn schema includes fields such as:
- `customer_id`
- `age`
- `gender`
- `region`
- `senior_citizen`
- `partner`
- `dependents`
- `tenure`
- `contract`
- `monthly_charges`
- `total_charges`
- `churn`

## Synthetic Data Generation
Run:
```bash
python scripts/generate_synthetic_customer_churn.py
```

This creates:
- `data/reference/reference.csv`
- `data/incoming/batch_001.csv`
- `data/incoming/batch_002.csv`
- `data/incoming/batch_003.csv`
- `data/incoming/batch_004.csv`

## Verification Commands
Run the data-side checks with:
```bash
python test_dataset_loader.py
python test_batch_manager.py
python test_schema.py
python test_preprocessor.py
python test_feature_alignment.py
```

## Output Artifacts
- `reports/predictions/` – per-batch prediction exports
- `reports/drift_reports/` – drift analysis outputs
- `reports/metrics/` – evaluation metrics logs
- `data/processed/` – processed data outputs when enabled

## Notes
This data engineering layer is designed to be reusable for other tabular classification use cases by changing config files and schema definitions rather than rewriting the pipeline logic.
