# ML Drift Detection and Retraining Pipeline

## Overview

This project is an end-to-end **ML Drift Detection and Retraining Pipeline** designed to monitor incoming datasets, validate schema consistency, preprocess data, detect drift against a reference dataset, and trigger retraining workflows when significant drift is found.

The system is built in a modular and configuration-driven way so that components such as schema validation, dataset loading, preprocessing, drift detection, and retraining can be managed independently and extended easily for real-world MLOps workflows.

---

## Problem Statement

Machine learning models often degrade in performance when the characteristics of incoming production data change over time. This issue is commonly known as **data drift** or **dataset drift**.

The goal of this project is to build a pipeline that:

* monitors new incoming data
* validates whether the dataset structure matches the expected schema
* preprocesses the data consistently
* compares new data with a baseline/reference dataset
* detects whether significant drift has occurred
* generates drift analysis outputs
* triggers retraining workflows when drift exceeds a defined threshold

---

## Key Features

* **Schema Validation**

  * Validates dataset columns, datatypes, and structure against predefined schema rules.
* **Config-Driven Pipeline**

  * Uses YAML-based configuration files for flexibility and reusability.
* **Dataset Loading and Batch Handling**

  * Loads training/reference/current datasets in a structured manner.
* **Preprocessing Module**

  * Handles transformations and prepares datasets for downstream drift analysis and model workflows.
* **Drift Detection**

  * Compares current data against reference data using statistical drift checks.
* **Logging**

  * Structured logging for easier debugging and pipeline tracking.
* **Retraining Trigger Framework**

  * Supports retraining workflows when drift is detected beyond threshold.
* **Modular Project Design**

  * Easy to maintain, test, and extend.

---

## Project Architecture

The pipeline follows a staged architecture:

1. **Load project configurations**
2. **Validate schema**
3. **Load reference and current datasets**
4. **Preprocess the datasets**
5. **Run drift detection**
6. **Generate drift results / reports**
7. **Trigger retraining if drift is detected**
8. **Store artifacts and logs**

---

## Project Structure

```bash
ml-drift-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── test_config.py
│
├── config/
│   ├── schema.yaml
│   ├── pipeline_config.yaml
│   └── model_config.yaml
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── schema_validation.py
│   │   ├── preprocessing.py
│   │   ├── drift_detection.py
│   │   └── retraining.py
│   │
│   ├── pipeline/
│   │   └── training_pipeline.py
│   │
│   ├── entity/
│   │   └── config_entity.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── common.py
│       └── config_loader.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── tests/
│   └── test_pipeline.py
│
├── artifacts/
│   ├── drift_reports/
│   ├── processed_data/
│   └── models/
│
└── sample_data/
    ├── reference_data.csv
    └── current_data.csv
```

> Note: The exact structure may vary slightly depending on your implementation.

---

## Tech Stack

* **Programming Language:** Python
* **Data Handling:** Pandas, NumPy
* **Machine Learning / Drift Utilities:** Scikit-learn, Evidently / custom statistical methods
* **Configuration Management:** YAML
* **Logging:** Python logging
* **Testing:** Pytest
* **Notebook Experiments:** Jupyter Notebook

---

## Workflow Explanation

### 1. Configuration Loading

The pipeline begins by loading all configuration files such as:

* schema configuration
* pipeline configuration
* model / drift configuration

This ensures that paths, thresholds, feature lists, and validation rules are centralized and easy to update.

---

### 2. Schema Validation

The schema validation module checks whether the incoming dataset matches the expected structure.

Typical checks include:

* required columns present
* column count validation
* datatype checks
* missing or unexpected columns

If schema validation fails, the pipeline can stop early and log the issue before drift analysis begins.

---

### 3. Dataset Loading

The ingestion module loads:

* **reference dataset** or baseline data
* **current dataset** or incoming batch data

These datasets are used for comparison in drift analysis.

---

### 4. Data Preprocessing

The preprocessing stage ensures that the input data is transformed consistently before drift checks or model training.

Possible preprocessing tasks include:

* handling missing values
* encoding categorical features
* scaling / normalization
* dropping irrelevant columns
* feature selection
* datatype conversion

---

### 5. Drift Detection

The drift detection component compares the incoming/current dataset with the reference dataset.

Depending on implementation, this may include:

* statistical tests for numerical features
* categorical distribution comparison
* feature-wise drift scoring
* overall dataset drift decision
* drift thresholds for alerting or retraining

The output of this stage determines whether the new data distribution has shifted enough to require intervention.

---

### 6. Drift Report Generation

After drift analysis, the system can generate:

* drift summary
* feature-wise drift status
* drift score outputs
* HTML / JSON / CSV reports
* logs for monitoring and debugging

---

### 7. Retraining Trigger

If drift exceeds the defined threshold, the pipeline can trigger a retraining workflow.

This retraining step may include:

* preparing updated training data
* retraining the model
* storing new model artifacts
* updating model version / deployment candidate

---

## Configuration Files

### `schema.yaml`

Contains schema-related rules such as:

* expected column names
* numerical / categorical columns
* target column
* validation constraints

### `pipeline_config.yaml`

Contains pipeline-level settings such as:

* artifact paths
* dataset paths
* report output locations
* pipeline execution settings

### `model_config.yaml`

Contains model / drift settings such as:

* drift threshold
* model training parameters
* preprocessing configuration
* retraining rules

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Run the main pipeline

```bash
python main.py
```

or if your entry file is different:

```bash
python test_config.py
```

or

```bash
python src/pipeline/training_pipeline.py
```

Use the command that matches your actual project entry point.

---

## Example Pipeline Flow

A typical execution flow of the project looks like this:

```python
load_all_configs()
validate_schema()
load_reference_dataset()
load_current_dataset()
preprocess_data()
run_drift_detection()
generate_drift_report()
trigger_retraining_if_needed()
```

---

## Sample Use Cases

This project can be extended for:

* production ML model monitoring
* batch-based drift analysis pipelines
* automated retraining workflows
* MLOps experimentation
* data quality and model health monitoring systems

---

## Outputs

Depending on implementation, the project can produce:

* drift reports
* processed datasets
* logs
* trained / retrained model artifacts
* validation summaries
* monitoring outputs

---

## Future Improvements

* Add **MLflow** for experiment tracking and model versioning
* Add **Docker** support for containerized deployment
* Add **CI/CD pipelines** for automated testing and deployment
* Add **Airflow / Prefect orchestration**
* Add **real-time drift monitoring dashboard**
* Add **email / Slack alerts** when drift is detected
* Integrate with **cloud storage and model registry**
* Add **unit tests and integration tests** for all components

---

## Key Learnings from the Project

* Building a modular ML pipeline using Python
* Designing config-driven workflows for scalability
* Implementing schema validation and preprocessing stages
* Understanding data drift and retraining strategies
* Structuring an MLOps-style project for maintainability and extension

---

## Author

**Niranjan M**
B.Tech CSE Student | Interested in Data Engineering, Machine Learning, and MLOps


---

## License

This project is intended for learning, experimentation, and portfolio demonstration.
You can add an MIT License if you want to make it open source.
