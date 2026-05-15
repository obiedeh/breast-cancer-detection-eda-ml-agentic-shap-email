from pathlib import Path

from medical_ai_explainability.explainability import FeatureContribution
from medical_ai_explainability.models import ModelResult
from medical_ai_explainability.reporting import write_reports


def test_report_generation(tmp_path: Path):
    result = ModelResult(
        name="logistic_regression",
        estimator=object(),
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
    )

    assert artifacts.model_report.exists()
    assert artifacts.explainability_report.exists()
    assert "human review" in artifacts.model_report.read_text(encoding="utf-8").lower()
