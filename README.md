# Agentic Medical AI Explainability

Explainable, reproducible medical AI workflow for model benchmarking, feature attribution, and human-review reports.

This repository is a focused engineering project, not a fake hospital platform. The included workflow uses the sklearn Wisconsin Breast Cancer dataset as a compact, deterministic benchmark for training, evaluating, explaining, and reporting on tabular classification models.

## Safety Boundary

This project is for engineering and research demonstration only.

It does not provide medical diagnosis, treatment instructions, autonomous clinical recommendations, or doctor replacement functionality. Model outputs and explanations are artifacts for qualified human review.

## What Works Now

- Load the sklearn breast cancer dataset
- Validate tabular schema and target values
- Create reproducible stratified train/test splits
- Train baseline classifiers
- Select a champion model by ROC AUC
- Calculate holdout classification metrics
- Generate global feature importance and local sample explanations
- Write markdown reports, a model card, machine-readable metrics, and SVG plots
- Run tests, linting, and type checks through standard Python tooling

## Repository Structure

```text
.
|-- configs/default.yaml
|-- docs/ARCHITECTURE.md
|-- reports/
|   |-- README.md
|   |-- generated/
|   |-- sample_explainability_report.md
|   `-- sample_model_report.md
|-- src/medical_ai_explainability/
|   |-- cli.py
|   |-- data.py
|   |-- evaluation.py
|   |-- explainability.py
|   |-- features.py
|   |-- models.py
|   |-- reporting.py
|   `-- schema.py
|-- tests/
|-- MODEL_CARD.md
|-- PORTFOLIO_DELIVERABLES.md
|-- Makefile
`-- pyproject.toml
```

## Quickstart

```bash
make install-dev
make test
make run-sample
```

On Windows PowerShell, use the virtual environment Python directly after installation:

```powershell
.\.venv\Scripts\python.exe -m medical_ai_explainability.cli run-sample --config configs/default.yaml
```

The sample workflow writes reproducible proof artifacts:

```text
reports/generated/model_report.md
reports/generated/explainability_report.md
reports/generated/model_card.md
reports/generated/metrics.json
reports/generated/confusion_matrix.svg
reports/generated/roc_curve.svg
reports/generated/feature_importance.svg
```

For the full local verification pass:

```bash
make verify
```

## Docker

Build and run the reproducible sample workflow:

```bash
docker build -t medical-ai-explainability:latest .
docker run --rm medical-ai-explainability:latest
```

Run the test suite inside the container:

```bash
docker run --rm medical-ai-explainability:latest python -m pytest -q
```

## Development Targets

```bash
make install      # install package
make install-dev  # install package with dev/explainability/notebook extras
make test         # run pytest
make run-sample   # run reproducible workflow and write reports
make lint         # run ruff
make typecheck    # run mypy
make verify       # lint, typecheck, test, run sample, validate artifacts
```

## Deliverables

See [PORTFOLIO_DELIVERABLES.md](PORTFOLIO_DELIVERABLES.md) for the credibility checklist and generated output map.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, limitations, evaluation approach, and safety boundaries.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the package map and Mermaid workflow diagram.

## Positioning

The repo focuses on explainable and operational medical AI workflows with human-in-the-loop review: reproducibility, transparent metrics, reviewable feature attributions, and report artifacts that keep clinical claims grounded.
