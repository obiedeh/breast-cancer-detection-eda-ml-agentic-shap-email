# Model Card

This model card documents the repository workflow and the generated champion model artifacts. It is for engineering and research demonstration only.

## Model Details

- Dataset: sklearn Wisconsin Breast Cancer dataset
- Task: binary classification benchmark
- Candidate models: logistic regression, random forest, RBF SVM
- Champion selection metric: ROC AUC on a deterministic holdout split
- Explainability: native or permutation global feature importance, plus local sample attribution

## Intended Use

This project demonstrates reproducible model training, evaluation, explainability, and reporting patterns for human-review AI workflows.

## Out of Scope

- Medical diagnosis
- Treatment recommendation
- Autonomous clinical decision-making
- Direct clinical deployment

## Evaluation

The sample pipeline reports accuracy, precision, recall, F1, ROC AUC, confusion matrix values, and ROC curve points from a held-out test split.

Generated outputs are written to `reports/generated/`.

## Safety and Review

All outputs require qualified human review. Feature attribution helps inspect model behavior, but it does not establish clinical causality or replace professional judgment.
