"""Command line interface for the reproducible sample workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import yaml

from medical_ai_explainability.data import load_breast_cancer_dataset, split_dataset
from medical_ai_explainability.explainability import global_feature_importance, local_explanation
from medical_ai_explainability.models import select_champion, train_baselines
from medical_ai_explainability.reporting import write_reports
from medical_ai_explainability.schema import TabularSchema, validate_schema


def run_sample(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Run the full reproducible workflow and write markdown reports."""

    config = _load_config(config_path)
    dataset = load_breast_cancer_dataset()
    validate_schema(
        dataset.features,
        dataset.target,
        TabularSchema(feature_names=dataset.feature_names),
    )
    split = split_dataset(
        dataset,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["random_state"]),
    )
    model_results = train_baselines(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        feature_names=dataset.feature_names,
        random_state=int(config["random_state"]),
    )
    champion = select_champion(model_results, metric=str(config["model_selection"]["metric"]))
    top_k = int(config["explainability"]["top_k"])
    global_importance = global_feature_importance(
        champion.estimator,
        split.X_test,
        split.y_test,
        dataset.feature_names,
        random_state=int(config["random_state"]),
        top_k=top_k,
    )
    sample_position = int(config["explainability"]["sample_index"])
    sample = split.X_test.iloc[sample_position]
    local = local_explanation(
        champion.estimator,
        sample,
        split.X_train,
        dataset.feature_names,
        top_k=top_k,
    )
    report_dir = output_dir or Path(config["reports"]["output_dir"])
    artifacts = write_reports(
        output_dir=report_dir,
        model_results=model_results,
        champion=champion,
        global_importance=global_importance,
        local_explanation=local,
        sample_id=str(sample.name),
        X_test=split.X_test,
        y_test=split.y_test,
    )
    return {
        "champion": champion.name,
        "metrics": champion.metrics,
        "model_report": str(artifacts.model_report),
        "explainability_report": str(artifacts.explainability_report),
        "model_card": str(artifacts.model_card),
        "metrics_json": str(artifacts.metrics_json),
        "confusion_matrix": str(artifacts.confusion_matrix_svg),
        "roc_curve": str(artifacts.roc_curve_svg),
        "feature_importance": str(artifacts.feature_importance_svg),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run explainable medical AI demo workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("run-sample", help="Train baselines and write reports.")
    sample.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    sample.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-sample":
        result = run_sample(args.config, args.output_dir)
        print(f"Champion model: {result['champion']}")
        print(f"ROC AUC: {result['metrics']['roc_auc']:.3f}")
        print(f"Model report: {result['model_report']}")
        print(f"Explainability report: {result['explainability_report']}")
        print(f"Model card: {result['model_card']}")
        print(f"Metrics JSON: {result['metrics_json']}")
        print(f"Confusion matrix: {result['confusion_matrix']}")
        print(f"ROC curve: {result['roc_curve']}")
        print(f"Feature importance: {result['feature_importance']}")
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_config(config, path)
    return cast(dict[str, Any], config)


_REQUIRED_CONFIG_KEYS: dict[str, list[str]] = {
    "root": ["random_state", "data", "model_selection", "explainability", "reports"],
    "data": ["test_size"],
    "model_selection": ["metric"],
    "explainability": ["top_k", "sample_index"],
    "reports": ["output_dir"],
}


def _validate_config(config: dict[str, Any], path: Path) -> None:
    for key in _REQUIRED_CONFIG_KEYS["root"]:
        if key not in config:
            raise ValueError(f"Config '{path}' is missing required key: '{key}'")
    for section in ("data", "model_selection", "explainability", "reports"):
        for key in _REQUIRED_CONFIG_KEYS[section]:
            if key not in config[section]:
                raise ValueError(f"Config '{path}' is missing '{section}.{key}'")


if __name__ == "__main__":
    raise SystemExit(main())
