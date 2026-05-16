# AGENTS.md — agentic-medical-ai-explainability

## Skill Selection

| Trigger | Skill |
|---|---|
| Model selection, champion logic, metric gaps | `production-architecture-reviewer` |
| SHAP, permutation importance, attribution drift | `observability-generator` |
| CI failures, import errors, test regressions | `runtime-stability-debugger` |
| Packaging, repo structure, dependency issues | `repo-hardening-refactor` |
| Physical AI safety patterns, human override | `physical-ai-safety-reviewer` |

## Project Structure

```
src/medical_ai_explainability/  ← all new code goes here
  schema.py         ← tabular schema validation
  data.py           ← dataset loading and splitting
  features.py       ← preprocessing components
  models.py         ← baseline training and champion selection
  evaluation.py     ← metric calculation
  explainability.py ← global and local feature attribution (SHAP + fallback)
  reporting.py      ← markdown report rendering
  cli.py            ← command-line workflow entry point
configs/            ← YAML configs (do not use config/ — it is empty and deprecated)
reports/generated/  ← auto-generated; do not hand-edit
tests/              ← pytest tests
```

## Rules for Codex

- All new modules go in `src/medical_ai_explainability/`. Do not create top-level scripts.
- Do not add training orchestration, MLflow, or experiment tracking unless explicitly requested.
- The `config/` directory is empty and deprecated — do not add files to it. Use `configs/`.
- SHAP is an optional extra (`pip install -e ".[explainability]"`). Any new explainability code must degrade gracefully when SHAP is unavailable.
- Do not remove or weaken the `REVIEW_NOTICE` human-review boundary in `reporting.py`.

## Medical AI Safety Rules

- Generated reports are **engineering artifacts**, not clinical decisions. The `REVIEW_NOTICE` in `reporting.py` is non-negotiable — do not remove it.
- Do not add autonomous decision output (alerts, treatment recommendations, risk scores) without explicit user instruction and a documented human-review gate.
- Credibility: do not claim model performance numbers in documentation without a matching artifact in `reports/`.

## Anti-Bloat

- Do not add abstract base classes or plugin registries for a three-model benchmark.
- Do not introduce feature flags, environment-based model switching, or a model registry.
- Do not add async, background workers, or serving infrastructure (FastAPI, Celery, etc.) without explicit request.
