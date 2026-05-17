# Model Card

This report is for engineering and research demonstration only. It does not provide a medical diagnosis, treatment instruction, or autonomous clinical recommendation. Outputs require qualified human review.

## Model Details

- Champion model: `logistic_regression`
- Dataset: sklearn Wisconsin Breast Cancer dataset
- Task: binary classification benchmark for explainability workflow validation
- Selection metric: ROC AUC

## Holdout Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.982 |
| Precision | 0.986 |
| Recall | 0.986 |
| F1 | 0.986 |
| ROC AUC | 0.996 |

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
