"""
Training utilities for the orchestration pipeline.
Automatically selects the best model.
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def train_baseline_model(X_ref, y_ref, metadata=None, model_config=None):

    X_train, X_val, y_train, y_val = train_test_split(
        X_ref,
        y_ref,
        test_size=0.2,
        random_state=42,
        stratify=y_ref
    )

    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=500,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        )
    }

    best_model = None
    best_score = -1
    best_name = ""

    print("\n========== Training Candidate Models ==========\n")

    for model_name, model in candidate_models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        score = f1_score(y_val, predictions)

        print(f"{model_name:<25} F1 Score : {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = model_name

    print("\n===============================================")
    print(f"Selected Model : {best_name}")
    print(f"Best F1 Score  : {best_score:.4f}")
    print("===============================================\n")

    return best_model


def retrain_model(X_ref, y_ref, X_batch, y_batch, metadata=None, model_config=None):

    X_new = np.vstack([X_ref, X_batch])
    y_new = np.concatenate([y_ref, y_batch])

    return train_baseline_model(
        X_ref=X_new,
        y_ref=y_new,
        metadata=metadata,
        model_config=model_config
    )