# Sample Explainability Report

This sample is for engineering and research demonstration only. It does not provide a medical diagnosis, treatment instruction, or autonomous clinical recommendation. Outputs require qualified human review.

## Global Feature Importance

| Feature | Importance | Method |
| --- | ---: | --- |
| worst concave points | 1.000000 | native |
| worst perimeter | 0.900000 | native |
| worst radius | 0.850000 | native |

## Local Explanation

| Feature | Sample Value | Contribution | Method |
| --- | ---: | ---: | --- |
| worst concave points | 0.123000 | 0.450000 | shap |
| worst perimeter | 103.400000 | -0.310000 | shap |
| worst radius | 16.200000 | 0.220000 | shap |

## Review Notes

Feature attribution can help reviewers inspect model behavior, but it does not establish clinical causality or replace qualified judgment.
