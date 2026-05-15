# Sample Model Evaluation Report

This sample is for engineering and research demonstration only. It does not provide a medical diagnosis, treatment instruction, or autonomous clinical recommendation. Outputs require qualified human review.

## Baseline Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.982 | 0.986 | 0.986 | 0.986 | 0.997 |
| random_forest | 0.956 | 0.958 | 0.972 | 0.965 | 0.995 |
| svm_rbf | 0.974 | 0.973 | 0.986 | 0.979 | 0.997 |

## Champion Model

- Selected model: `logistic_regression`
- Selection metric: ROC AUC

## Human Review Boundary

Model output is a decision-support artifact for review workflows. It is not a standalone clinical decision.
