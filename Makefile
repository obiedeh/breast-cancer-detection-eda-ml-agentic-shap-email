.PHONY: install install-dev test run-sample lint typecheck validate-artifacts verify

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.venv:
	python3 -m venv .venv

install: .venv
	$(PIP) install -e .

install-dev: .venv
	$(PIP) install -e ".[dev,explainability,notebook]"

test: .venv
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

run-sample: .venv
	$(PYTHON) -m medical_ai_explainability.cli run-sample --config configs/default.yaml

lint: .venv
	$(PYTHON) -m ruff check src tests

typecheck: .venv
	$(PYTHON) -m mypy src/medical_ai_explainability

validate-artifacts:
	test -s reports/generated/model_report.md
	test -s reports/generated/explainability_report.md
	test -s reports/generated/model_card.md
	test -s reports/generated/metrics.json
	test -s reports/generated/confusion_matrix.svg
	test -s reports/generated/roc_curve.svg
	test -s reports/generated/feature_importance.svg

verify: lint typecheck test run-sample validate-artifacts
