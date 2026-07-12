"""Result audit: machine checks over run manifests, metrics, and predictions.

Findings are returned as structured dicts and written to artifacts/ by
scripts/audit_results.py. A run passes only if every check passes.
"""

from __future__ import annotations

import json
from pathlib import Path

from signal_aug.evaluation.metrics import METRIC_NAMES
from signal_aug.experiments.manifest import validate_manifest


def _check_metrics_file(manifest: dict) -> list[str]:
    problems = []
    path = Path(manifest["metrics_path"])
    if not path.exists():
        return [f"metrics file missing: {path}"]
    metrics = json.loads(path.read_text())
    for name in METRIC_NAMES:
        value = metrics.get(name)
        if value is None:
            problems.append(f"metric missing: {name}")
        elif not (0.0 <= float(value) <= 1.0):
            problems.append(f"metric out of range: {name}={value}")
    return problems


def _check_predictions_file(manifest: dict) -> list[str]:
    path = Path(manifest["predictions_path"])
    if not path.exists():
        return [f"predictions file missing: {path}"]
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return [f"predictions file empty: {path}"]
    if lines[0] != "index,y_true,y_pred":
        return [f"predictions header unexpected: {lines[0]!r}"]
    metrics = json.loads(Path(manifest["metrics_path"]).read_text())
    n_expected = metrics.get("n_test")
    if n_expected is not None and len(lines) - 1 != n_expected:
        return [f"prediction rows ({len(lines) - 1}) != n_test ({n_expected})"]
    return []


def audit_run(manifest: dict) -> dict:
    problems = validate_manifest(manifest)
    if manifest.get("status") == "completed" and not problems:
        problems += _check_metrics_file(manifest)
        problems += _check_predictions_file(manifest)
    return {
        "run_id": manifest.get("run_id", "unknown"),
        "status": manifest.get("status"),
        "passed": not problems and manifest.get("status") == "completed",
        "problems": problems,
    }


def audit_all(manifests_dir: str | Path = "runs/manifests") -> dict:
    manifests_dir = Path(manifests_dir)
    results = []
    for path in sorted(manifests_dir.glob("*.json")):
        results.append(audit_run(json.loads(path.read_text())))
    return {
        "n_runs": len(results),
        "n_passed": sum(1 for r in results if r["passed"]),
        "n_failed_runs": sum(1 for r in results if r["status"] == "failed"),
        "n_problem_runs": sum(1 for r in results if r["problems"]),
        "runs": results,
    }
