# Agentic Medical AI Explainability

Explainable and operational medical AI workflows with human-in-the-loop review.

This repository is structured as a production-style Python package for reproducible model training, evaluation, explainability, and markdown reporting. The included workflow uses the sklearn Wisconsin Breast Cancer dataset as a compact engineering demonstration.

## Safety Boundary

This project is for engineering and research demonstration only.

It does not provide medical diagnosis, treatment instructions, autonomous clinical recommendations, or doctor replacement functionality. Model outputs and explanations are artifacts for qualified human review.

## Capabilities

- Load the sklearn breast cancer dataset
- Validate tabular schema and target values
- Create reproducible stratified train/test splits
- Train baseline classifiers
- Select a champion model by ROC AUC
- Calculate standard classification metrics
- Generate global feature importance and local sample explanations
- Write markdown reports to `reports/`

## Repository Structure

```text
.
├── configs/default.yaml
├── docs/ARCHITECTURE.md
├── reports/
│   ├── README.md
│   ├── sample_explainability_report.md
│   └── sample_model_report.md
├── src/medical_ai_explainability/
│   ├── cli.py
│   ├── data.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── features.py
│   ├── models.py
│   ├── reporting.py
│   └── schema.py
├── tests/
├── Makefile
└── pyproject.toml
```

## Quickstart

```bash
make install-dev
make test
make run-sample
```

The sample workflow writes:

```text
reports/generated/model_report.md
reports/generated/explainability_report.md
```

You can also run the CLI directly:

```bash
medical-ai-explainability run-sample --config configs/default.yaml
```

## Development Targets

```bash
make install      # install package
make install-dev  # install package with dev/explainability/notebook extras
make test         # run pytest
make run-sample   # run reproducible workflow and write reports
make lint         # run ruff
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the package map and Mermaid workflow diagram.

## Positioning

The repo focuses on explainable and operational medical AI workflows with human-in-the-loop review: reproducibility, transparent metrics, reviewable feature attributions, and report artifacts that keep clinical claims grounded.
