.PHONY: setup test smoke phase1 download aggregate audit report validate all-report clean-caches

setup:
	uv sync
	cd report && npm install

test:
	uv run pytest -q

smoke:
	uv run python scripts/run_experiment.py --config config/experiments/smoke.yaml

download:
	uv run python scripts/download_datasets.py

phase1:
	uv run python scripts/run_experiment.py --config config/experiments/phase1.yaml

phase2:
	uv run python scripts/run_experiment.py --config config/experiments/phase2.yaml

aggregate:
	uv run python scripts/aggregate_results.py

audit:
	uv run python scripts/audit_results.py

report:
	uv run python scripts/build_report.py

validate:
	uv run python scripts/validate_artifacts.py

# audit -> aggregate -> build report in one shot
all-report: audit aggregate report

clean-caches:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} +
