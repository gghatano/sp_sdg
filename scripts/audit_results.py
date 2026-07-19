#!/usr/bin/env python
"""Audit all run manifests and write artifacts/audit_report.json."""

import json
from pathlib import Path

from signal_aug.evaluation.audit import audit_all


def main() -> None:
    report = audit_all("runs/manifests")
    out = Path("artifacts/audit_report.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[audit] {report['n_passed']}/{report['n_runs']} passed, "
          f"{report['n_failed_runs']} failed runs, {report['n_problem_runs']} with problems")
    for run in report["runs"]:
        for problem in run["problems"]:
            print(f"  [problem] {run['run_id']}: {problem}")
    if report["n_problem_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
