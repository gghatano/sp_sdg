#!/usr/bin/env python
"""Aggregate manifests/metrics into report/assets/data/results.json."""

from signal_aug.reporting.aggregate import build_results_json


def main() -> None:
    data = build_results_json()
    print(f"[aggregate] {len(data['runs'])} runs -> report/assets/data/results.json "
          f"({len(data['summary'])} summary rows, {len(data['failed_runs'])} failed)")


if __name__ == "__main__":
    main()
