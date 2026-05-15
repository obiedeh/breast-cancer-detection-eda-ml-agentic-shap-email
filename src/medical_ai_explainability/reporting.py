"""Markdown reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medical_ai_explainability.explainability import FeatureContribution
from medical_ai_explainability.models import ModelResult

REVIEW_NOTICE = (
    "This report is for engineering and research demonstration only. "
    "It does not provide a medical diagnosis, treatment instruction, or autonomous "
    "clinical recommendation. Outputs require qualified human review."
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths written by the report generator."""

    model_report: Path
    explainability_report: Path


def write_reports(
    *,
    output_dir: Path,
    model_results: list[ModelResult],
    champion: ModelResult,
    global_importance: list[FeatureContribution],
    local_explanation: list[FeatureContribution],
    sample_id: str,
) -> ReportArtifacts:
    """Write model and explainability reports to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_report.md"
    explainability_path = output_dir / "explainability_report.md"
    model_path.write_text(
        render_model_report(model_results=model_results, champion=champion),
        encoding="utf-8",
    )
    explainability_path.write_text(
        render_explainability_report(
            champion=champion,
            global_importance=global_importance,
            local_explanation=local_explanation,
            sample_id=sample_id,
        ),
        encoding="utf-8",
    )
    return ReportArtifacts(model_report=model_path, explainability_report=explainability_path)


def render_model_report(*, model_results: list[ModelResult], champion: ModelResult) -> str:
    rows = "\n".join(
        (
            "| {name} | {accuracy:.3f} | {precision:.3f} | {recall:.3f} | "
            "{f1:.3f} | {roc_auc:.3f} |"
        ).format(
            name=result.name,
            **result.metrics,
        )
        for result in model_results
    )
    return f"""# Model Evaluation Report

{REVIEW_NOTICE}

## Dataset

- Source: sklearn Wisconsin Breast Cancer dataset
- Intended use: reproducible engineering workflow demonstration
- Target labels: encoded dataset classes, used only for model benchmarking

## Baseline Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

## Champion Model

- Selected model: `{champion.name}`
- Selection metric: ROC AUC
- ROC AUC: {champion.metrics["roc_auc"]:.3f}

## Human Review Boundary

The model output is a decision-support artifact for review workflows.
It is not a standalone clinical decision.
"""


def render_explainability_report(
    *,
    champion: ModelResult,
    global_importance: list[FeatureContribution],
    local_explanation: list[FeatureContribution],
    sample_id: str,
) -> str:
    global_rows = "\n".join(
        f"| {item.feature} | {item.contribution:.6f} | {item.method} |"
        for item in global_importance
    )
    local_rows = "\n".join(
        f"| {item.feature} | {item.value:.6f} | {item.contribution:.6f} | {item.method} |"
        for item in local_explanation
    )
    return f"""# Explainability Report

{REVIEW_NOTICE}

## Champion

- Model: `{champion.name}`
- Explanation scope: feature attribution for engineering inspection

## Global Feature Importance

| Feature | Importance | Method |
| --- | ---: | --- |
{global_rows}

## Local Explanation

- Sample ID: `{sample_id}`

| Feature | Sample Value | Contribution | Method |
| --- | ---: | ---: | --- |
{local_rows}

## Review Notes

Feature attribution can help reviewers inspect model behavior.
It does not establish clinical causality or replace qualified judgment.
"""
