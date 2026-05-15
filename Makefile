.PHONY: install install-dev test run-sample lint

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
	.venv/bin/ruff check src tests
