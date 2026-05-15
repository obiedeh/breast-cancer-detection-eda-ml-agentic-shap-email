# Explainability Report

This report is for engineering and research demonstration only. It does not provide a medical diagnosis, treatment instruction, or autonomous clinical recommendation. Outputs require qualified human review.

## Champion

- Model: `logistic_regression`
- Explanation scope: feature attribution for engineering inspection

## Global Feature Importance

| Feature | Importance | Method |
| --- | ---: | --- |
| worst texture | 1.242272 | native |
| radius error | 1.087929 | native |
| worst area | 0.979282 | native |
| area error | 0.958096 | native |
| worst radius | 0.946000 | native |
| worst concave points | 0.945296 | native |
| worst symmetry | 0.928729 | native |
| worst concavity | 0.827180 | native |

## Local Explanation

- Sample ID: `256`

| Feature | Sample Value | Contribution | Method |
| --- | ---: | ---: | --- |
| worst area | 1926.000000 | 1035.075232 | centered_native_importance |
| mean area | 1207.000000 | 306.130783 | centered_native_importance |
| area error | 106.400000 | 64.342122 | centered_native_importance |
| worst perimeter | 178.600000 | 55.046754 | centered_native_importance |
| mean perimeter | 133.600000 | 19.381131 | centered_native_importance |
| worst texture | 36.270000 | 13.196289 | centered_native_importance |
| worst radius | 25.050000 | 8.393648 | centered_native_importance |
| mean texture | 28.770000 | 5.296533 | centered_native_importance |

## Review Notes

Feature attribution can help reviewers inspect model behavior.
It does not establish clinical causality or replace qualified judgment.
