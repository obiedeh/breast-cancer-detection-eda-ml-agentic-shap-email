"""Report and evidence artifact utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from medical_ai_explainability.explainability import FeatureContribution
from medical_ai_explainability.models import ModelResult, predict_positive_class_probability

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
    model_card: Path
    metrics_json: Path
    confusion_matrix_svg: Path
    roc_curve_svg: Path
    feature_importance_svg: Path


def write_reports(
    *,
    output_dir: Path,
    model_results: list[ModelResult],
    champion: ModelResult,
    global_importance: list[FeatureContribution],
    local_explanation: list[FeatureContribution],
    sample_id: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ReportArtifacts:
    """Write markdown reports and proof artifacts to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_report.md"
    explainability_path = output_dir / "explainability_report.md"
    model_card_path = output_dir / "model_card.md"
    metrics_path = output_dir / "metrics.json"
    confusion_path = output_dir / "confusion_matrix.svg"
    roc_path = output_dir / "roc_curve.svg"
    importance_path = output_dir / "feature_importance.svg"

    estimator: Any = champion.estimator
    y_pred = pd.Series(estimator.predict(X_test), index=y_test.index)
    y_score = predict_positive_class_probability(champion.estimator, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_score)

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
    model_card_path.write_text(render_model_card(champion=champion), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "champion": champion.name,
                "metrics": champion.metrics,
                "confusion_matrix": {
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                },
                "dataset": "sklearn_breast_cancer",
                "safety_boundary": "engineering demonstration; qualified human review required",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    confusion_path.write_text(
        render_confusion_matrix_svg(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        encoding="utf-8",
    )
    roc_path.write_text(
        render_roc_curve_svg(fpr=[float(x) for x in fpr], tpr=[float(y) for y in tpr]),
        encoding="utf-8",
    )
    importance_path.write_text(
        render_feature_importance_svg(global_importance),
        encoding="utf-8",
    )
    return ReportArtifacts(
        model_report=model_path,
        explainability_report=explainability_path,
        model_card=model_card_path,
        metrics_json=metrics_path,
        confusion_matrix_svg=confusion_path,
        roc_curve_svg=roc_path,
        feature_importance_svg=importance_path,
    )


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


def render_model_card(*, champion: ModelResult) -> str:
    return f"""# Model Card

{REVIEW_NOTICE}

## Model Details

- Champion model: `{champion.name}`
- Dataset: sklearn Wisconsin Breast Cancer dataset
- Task: binary classification benchmark for explainability workflow validation
- Selection metric: ROC AUC

## Holdout Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | {champion.metrics["accuracy"]:.3f} |
| Precision | {champion.metrics["precision"]:.3f} |
| Recall | {champion.metrics["recall"]:.3f} |
| F1 | {champion.metrics["f1"]:.3f} |
| ROC AUC | {champion.metrics["roc_auc"]:.3f} |

## Intended Use

This model is intended for reproducible engineering demonstration, report generation, and
explainability review patterns.

## Out of Scope

- Medical diagnosis
- Treatment recommendation
- Autonomous clinical decision-making
- Deployment in clinical care

## Required Review

Any model output must be reviewed by qualified humans before being interpreted in a
real-world context.
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


def render_confusion_matrix_svg(*, tn: int, fp: int, fn: int, tp: int) -> str:
    max_count = max(tn, fp, fn, tp, 1)

    def color(value: int) -> str:
        intensity = 255 - int(150 * (value / max_count))
        return f"rgb({intensity},{intensity},255)"

    cells = [
        (80, 90, "TN", tn),
        (230, 90, "FP", fp),
        (80, 210, "FN", fn),
        (230, 210, "TP", tp),
    ]
    rects = "\n".join(
        (
            f'<rect x="{x}" y="{y}" width="130" height="90" '
            f'fill="{color(value)}" stroke="#1f2937" />'
            f'<text x="{x + 65}" y="{y + 38}" '
            f'text-anchor="middle" font-size="18">{label}</text>'
            f'<text x="{x + 65}" y="{y + 64}" text-anchor="middle" '
            f'font-size="22" font-weight="bold">{value}</text>'
        )
        for x, y, label, value in cells
    )
    return f"""{_svg_open(width=420, height=340)}
<rect width="420" height="340" fill="white" />
<text x="210" y="32" text-anchor="middle" font-size="22" font-weight="bold">Confusion Matrix</text>
<text x="210" y="320" text-anchor="middle" font-size="14">Predicted label</text>
<text x="22" y="175" text-anchor="middle" font-size="14" transform="rotate(-90 22 175)">
Actual label</text>
<text x="145" y="75" text-anchor="middle" font-size="13">Pred 0</text>
<text x="295" y="75" text-anchor="middle" font-size="13">Pred 1</text>
<text x="58" y="140" text-anchor="middle" font-size="13">Actual 0</text>
<text x="58" y="260" text-anchor="middle" font-size="13">Actual 1</text>
{rects}
</svg>
"""


def render_roc_curve_svg(*, fpr: list[float], tpr: list[float]) -> str:
    width = 420
    height = 320
    left = 60
    top = 40
    plot_width = 310
    plot_height = 230
    points = " ".join(
        f"{left + x * plot_width:.2f},{top + (1 - y) * plot_height:.2f}"
        for x, y in zip(fpr, tpr, strict=False)
    )
    return f"""{_svg_open(width=width, height=height)}
<rect width="{width}" height="{height}" fill="white" />
<text x="210" y="26" text-anchor="middle" font-size="22" font-weight="bold">ROC Curve</text>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}"
y2="{top + plot_height}" stroke="#111827" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" />
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top}"
stroke="#9ca3af" stroke-dasharray="5 5" />
<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3" />
<text x="215" y="305" text-anchor="middle" font-size="14">False positive rate</text>
<text x="18" y="155" text-anchor="middle" font-size="14" transform="rotate(-90 18 155)">
True positive rate</text>
</svg>
"""


def render_feature_importance_svg(global_importance: list[FeatureContribution]) -> str:
    width = 760
    row_height = 34
    top = 54
    left = 245
    bar_width = 450
    height = top + row_height * len(global_importance) + 32
    max_value = max((abs(item.contribution) for item in global_importance), default=1.0)
    rows = []
    for index, item in enumerate(global_importance):
        y = top + index * row_height
        bar = 0 if max_value == 0 else abs(item.contribution) / max_value * bar_width
        rows.append(
            f'<text x="{left - 12}" y="{y + 21}" text-anchor="end" '
            f'font-size="13">{_escape_svg(item.feature)}</text>'
            f'<rect x="{left}" y="{y + 6}" width="{bar:.2f}" '
            f'height="20" fill="#2563eb" />'
            f'<text x="{left + bar + 8}" y="{y + 21}" '
            f'font-size="12">{item.contribution:.3f}</text>'
        )
    return f"""{_svg_open(width=width, height=height)}
<rect width="{width}" height="{height}" fill="white" />
<text x="380" y="28" text-anchor="middle" font-size="22" font-weight="bold">
Global Feature Importance</text>
{''.join(rows)}
</svg>
"""


def _svg_open(*, width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def _escape_svg(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
