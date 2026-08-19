"""Aggregate run manifests + metrics into report input data.

Output: report/assets/data/results.json - the single data source for the
HTML report (no results are ever hand-typed into HTML; spec sections 3.10, 9).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


# Auxiliary/methodological experiments excluded from the study report summary
# (they duplicate the none baseline for side-analyses, e.g. the H2 gap study).
AUX_RUN_PREFIXES = ("h2gap_",)


def collect_runs(manifests_dir: str | Path = "runs/manifests") -> list[dict]:
    rows = []
    for path in sorted(Path(manifests_dir).glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["run_id"].startswith(AUX_RUN_PREFIXES):
            continue
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
                row.update(json.loads(metrics_path.read_text(encoding="utf-8")))
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


def learning_curves(summary: list[dict]) -> dict:
    """Group summary rows into per-(dataset, model, augmentation) accuracy
    curves over train_fraction, for the Phase 2 learning-curve figures."""
    curves: dict[str, list[dict]] = {}
    for s in summary:
        key = f"{s['dataset']}|{s['model']}|{s['augmentation']}"
        curves.setdefault(key, []).append(
            {
                "train_fraction": s.get("train_fraction", 1.0),
                "accuracy_mean": s["accuracy_mean"],
                "accuracy_std": s["accuracy_std"],
                "macro_f1_mean": s["macro_f1_mean"],
            }
        )
    for points in curves.values():
        points.sort(key=lambda p: p["train_fraction"])
    return curves


def _subject_metric_rows(manifests_dir: Path, prefix: str = "subject_count_") -> list[dict]:
    """Subject-count rows carry subject_count in the manifest, not the metrics
    file, so join them here for the reduction analysis. `prefix` selects the
    dataset's run family (e.g. "subject_count_" for UCI HAR, "wisdm_subject_count_"
    for WISDM); the two families never overlap by run_id."""
    rows = []
    for path in sorted(Path(manifests_dir).glob(f"{prefix}*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("status") != "completed" or not m.get("metrics_path"):
            continue
        metrics_path = Path(m["metrics_path"])
        if not metrics_path.exists():
            continue
        row = {
            "phase": m["phase"],
            "augmentation": m["augmentation"],
            "model": m["model"],
            "subject_count": m.get("subject_count"),
            "status": "completed",
        }
        row.update(json.loads(metrics_path.read_text(encoding="utf-8")))
        rows.append(row)
    return rows


def build_results_json(
    manifests_dir: str | Path = "runs/manifests",
    audit_path: str | Path = "artifacts/audit_report.json",
    output_path: str | Path = "report/assets/data/results.json",
    config_dir: str | Path = "config",
) -> dict:
    import yaml

    from signal_aug.evaluation.reduction import reduction_analysis
    from signal_aug.evaluation.stats import wilcoxon_vs_none

    rows = collect_runs(manifests_dir)
    summary = summarize(rows)
    audit = None
    audit_path = Path(audit_path)
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit.pop("runs", None)  # keep the report data compact

    # Wilcoxon test is meaningful once fractions/datasets give >=5 pairs (Phase 2)
    phase2_rows = [r for r in rows if r.get("phase") == 2]
    stats = wilcoxon_vs_none(phase2_rows) if phase2_rows else []

    # Phase 4-5: subject-count reduction analysis, target read from the
    # pre-registered config (never chosen post-hoc; spec section 8)
    def _reduction_for(prefix: str, cfg_name: str):
        rows_s = _subject_metric_rows(Path(manifests_dir), prefix=prefix)
        cfg_path = Path(config_dir) / "experiments" / cfg_name
        if not (rows_s and cfg_path.exists()):
            return None
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        red = reduction_analysis(
            rows_s, target=float(cfg["target_value"]), metric=cfg["target_metric"]
        )
        red["pre_registered"] = bool(cfg.get("pre_registered"))
        red["dataset"] = cfg.get("dataset")
        return red

    # UCI HAR (primary, Phase 4-5) and WISDM (external validity, issue #21 DS-1)
    reduction = _reduction_for("subject_count_", "subject_count.yaml")
    reduction_wisdm = _reduction_for("wisdm_subject_count_", "wisdm_subject_count.yaml")

    data = {
        "runs": rows,
        "summary": summary,
        "learning_curves": learning_curves([s for s in summary if s["dataset"] != "synthetic"]),
        "stats": stats,
        "reduction": reduction,
        "reduction_wisdm": reduction_wisdm,
        "failed_runs": [r for r in rows if r["status"] == "failed"],
        "audit": audit,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data
