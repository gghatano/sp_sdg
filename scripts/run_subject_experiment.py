#!/usr/bin/env python
"""Run the subject-count learning-curve experiment (Phase 4).

Usage: uv run python scripts/run_subject_experiment.py --config config/experiments/subject_count.yaml
"""

import argparse

from signal_aug.experiments.subject_runner import run_subject_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    results = run_subject_experiment(args.config, resume=not args.no_resume)
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
