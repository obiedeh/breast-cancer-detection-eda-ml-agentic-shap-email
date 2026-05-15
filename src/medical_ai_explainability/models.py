"""Baseline model training and champion selection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from medical_ai_explainability.evaluation import calculate_metrics
from medical_ai_explainability.features import numeric_scaler


@dataclass(frozen=True)
class ModelResult:
    """Fitted model plus holdout metrics."""

    name: str
    estimator: object
    metrics: dict[str, float]


def build_baseline_models(feature_names: list[str], random_state: int) -> dict[str, object]:
    """Create deterministic baseline estimators for comparison."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", numeric_scaler(feature_names)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("preprocess", numeric_scaler(feature_names)),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def train_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    feature_names: list[str],
    random_state: int,
) -> list[ModelResult]:
    """Fit baseline models and evaluate them on the holdout set."""

    results: list[ModelResult] = []
    for name, estimator in build_baseline_models(feature_names, random_state).items():
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_score = predict_positive_class_probability(estimator, X_test)
        metrics = calculate_metrics(y_test, y_pred, y_score)
        results.append(ModelResult(name=name, estimator=estimator, metrics=metrics))
    return results


def predict_positive_class_probability(estimator: object, features: pd.DataFrame) -> pd.Series:
    """Return probability-like scores for target class 1 when available."""

    if hasattr(estimator, "predict_proba"):
        return pd.Series(estimator.predict_proba(features)[:, 1], index=features.index)
    if hasattr(estimator, "decision_function"):
        return pd.Series(estimator.decision_function(features), index=features.index)
    return pd.Series(estimator.predict(features), index=features.index)


def select_champion(results: list[ModelResult], metric: str = "roc_auc") -> ModelResult:
    """Select the model with the highest requested metric."""

    if not results:
        raise ValueError("No model results provided")
    return max(results, key=lambda result: result.metrics[metric])
