#!/usr/bin/env python
"""Run an experiment grid from a config file (resume-aware).

Usage: uv run python scripts/run_experiment.py --config config/experiments/smoke.yaml
"""

import argparse

from signal_aug.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true", help="re-run even completed runs")
    args = parser.parse_args()
    results = run_experiment(args.config, resume=not args.no_resume)
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
