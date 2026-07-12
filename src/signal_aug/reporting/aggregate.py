"""Aggregate run manifests + metrics into report input data.

Output: report/assets/data/results.json - the single data source for the
HTML report (no results are ever hand-typed into HTML; spec sections 3.10, 9).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def collect_runs(manifests_dir: str | Path = "runs/manifests") -> list[dict]:
    rows = []
    for path in sorted(Path(manifests_dir).glob("*.json")):
        manifest = json.loads(path.read_text())
        row = {
            "run_id": manifest["run_id"],
            "phase": manifest["phase"],
            "dataset": manifest["dataset"],
            "augmentation": manifest["augmentation"],
            "model": manifest["model"],
            "seed": manifest["seed"],
            "status": manifest["status"],
            "git_commit": manifest.get("git_commit", "")[:12],
            "git_dirty": manifest.get("git_dirty"),
            "ended_at": manifest.get("ended_at"),
            "python_version": manifest.get("python_version"),
            "train_fraction": manifest.get("train_fraction", 1.0),
        }
        if manifest["status"] == "completed" and manifest.get("metrics_path"):
            metrics_path = Path(manifest["metrics_path"])
            if metrics_path.exists():
                row.update(json.loads(metrics_path.read_text()))
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Mean/std across seeds for each (dataset, augmentation, model)."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["status"] != "completed" or "accuracy" not in row:
            continue
        key = (row["dataset"], row.get("train_fraction", 1.0), row["augmentation"], row["model"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (dataset, fraction, aug, model), members in sorted(groups.items()):
        entry = {
            "dataset": dataset,
            "train_fraction": fraction,
            "augmentation": aug,
            "model": model,
            "n_seeds": len(members),
        }
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = [m[metric] for m in members]
            entry[f"{metric}_mean"] = round(statistics.mean(values), 4)
            entry[f"{metric}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        summary.append(entry)
    return summary


def build_results_json(
    manifests_dir: str | Path = "runs/manifests",
    audit_path: str | Path = "artifacts/audit_report.json",
    output_path: str | Path = "report/assets/data/results.json",
) -> dict:
    rows = collect_runs(manifests_dir)
    audit = None
    audit_path = Path(audit_path)
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        audit.pop("runs", None)  # keep the report data compact
    data = {
        "runs": rows,
        "summary": summarize(rows),
        "failed_runs": [r for r in rows if r["status"] == "failed"],
        "audit": audit,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data
