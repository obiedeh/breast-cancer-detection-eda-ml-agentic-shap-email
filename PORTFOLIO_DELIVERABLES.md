# Portfolio Deliverables

This repository is scoped to a narrow, credible deliverable: a deterministic medical AI explainability workflow with reproducible proof artifacts.

## One-Command Checks

```bash
make test
make run-sample
```

The CI workflow also runs linting, type checks, and pytest.

## Proof Artifacts

| Artifact | Purpose |
| --- | --- |
| `reports/generated/model_report.md` | Champion model summary and baseline comparison |
| `reports/generated/explainability_report.md` | Global and local feature attribution report |
| `reports/generated/model_card.md` | Generated run-specific model card |
| `reports/generated/metrics.json` | Machine-readable metrics and confusion matrix |
| `reports/generated/confusion_matrix.svg` | Visual classification outcome summary |
| `reports/generated/roc_curve.svg` | Visual ROC curve proof artifact |
| `reports/generated/feature_importance.svg` | Visual global feature importance artifact |

## Credibility Boundary

The repo proves reproducible engineering behavior: data loading, schema validation, deterministic splitting, model comparison, explainability, reports, tests, and generated outputs.

It does not claim clinical validity, clinical deployment readiness, or autonomous medical decision-making.
