# Model Evaluation Report

This report is for engineering and research demonstration only. It does not provide a medical diagnosis, treatment instruction, or autonomous clinical recommendation. Outputs require qualified human review.

## Dataset

- Source: sklearn Wisconsin Breast Cancer dataset
- Intended use: reproducible engineering workflow demonstration
- Target labels: encoded dataset classes, used only for model benchmarking

## Baseline Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.982 | 0.986 | 0.986 | 0.986 | 0.996 |
| random_forest | 0.956 | 0.959 | 0.972 | 0.966 | 0.994 |
| svm_rbf | 0.982 | 0.986 | 0.986 | 0.986 | 0.995 |

## Champion Model

- Selected model: `logistic_regression`
- Selection metric: ROC AUC
- ROC AUC: 0.996

## Human Review Boundary

The model output is a decision-support artifact for review workflows.
It is not a standalone clinical decision.
