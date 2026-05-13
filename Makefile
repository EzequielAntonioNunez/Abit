.PHONY: install reinstall test lint format clean exp_001 exp_002 exp_002b

install:
	@if [ -d .venv ]; then \
		echo ".venv ya existe, reutilizando (usar make reinstall para recrear)"; \
	else \
		uv venv; \
	fi
	uv pip install -e ".[dev]"

reinstall:
	uv venv --clear
	uv pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src/ tests/ experiments/
	mypy src/

format:
	ruff format src/ tests/ experiments/
	ruff check --fix src/ tests/ experiments/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

exp_001:
	python experiments/exp_001_shannon_baseline/run.py

exp_002:
	python experiments/exp_002_activation_drift_last/run.py

exp_002b:
	python experiments/exp_002b_drift_decomposition/run.py
