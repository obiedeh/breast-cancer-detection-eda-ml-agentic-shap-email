from pathlib import Path

import numpy as np
import pandas as pd

from medical_ai_explainability.explainability import FeatureContribution
from medical_ai_explainability.models import ModelResult
from medical_ai_explainability.reporting import write_reports


class PredictableEstimator:
    def predict(self, X: pd.DataFrame):
        return [0, 1, 1, 0][: len(X)]

    def predict_proba(self, X: pd.DataFrame):
        return np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]][: len(X)])


def test_report_generation(tmp_path: Path):
    result = ModelResult(
        name="logistic_regression",
        estimator=PredictableEstimator(),
        metrics={
            "accuracy": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "f1": 0.9,
            "roc_auc": 0.95,
        },
    )
    contribution = FeatureContribution(
        feature="mean radius",
        value=12.3,
        contribution=0.5,
        method="native",
    )

    artifacts = write_reports(
        output_dir=tmp_path,
        model_results=[result],
        champion=result,
        global_importance=[contribution],
        local_explanation=[contribution],
        sample_id="sample-1",
        X_test=pd.DataFrame({"mean radius": [1.0, 2.0, 3.0, 4.0]}),
        y_test=pd.Series([0, 1, 1, 0]),
    )

    assert artifacts.model_report.exists()
    assert artifacts.explainability_report.exists()
    assert artifacts.model_card.exists()
    assert artifacts.metrics_json.exists()
    assert artifacts.confusion_matrix_svg.exists()
    assert artifacts.roc_curve_svg.exists()
    assert artifacts.feature_importance_svg.exists()
    assert "human review" in artifacts.model_report.read_text(encoding="utf-8").lower()
