#!/usr/bin/env python
"""H2 analysis (issue #12): SDG effect vs no-augmentation generalization gap.

Reads the h2gap none-baseline runs (with train_accuracy) and the aggregated
Phase 2 summary, then writes artifacts/h2_gap_analysis.json.
"""

import json
from pathlib import Path


def _collect_none_rows(manifests_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(manifests_dir.glob("h2gap_*.json")):
        m = json.loads(path.read_text())
        if m.get("status") != "completed" or not m.get("metrics_path"):
            continue
        mp = Path(m["metrics_path"])
        if not mp.exists():
            continue
        row = {"dataset": m["dataset"], "model": m["model"], "augmentation": m["augmentation"],
               "status": "completed"}
        row.update(json.loads(mp.read_text()))
        rows.append(row)
    return rows


def main() -> None:
    from signal_aug.evaluation.gap_analysis import analyze_gap

    none_rows = _collect_none_rows(Path("runs/manifests"))
    results = json.loads(Path("report/assets/data/results.json").read_text())
    summary = results.get("summary", [])
    analysis = analyze_gap(none_rows, summary, train_fraction=1.0)

    out = Path("artifacts/h2_gap_analysis.json")
    out.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    print(f"[h2] {analysis['n_points']} (dataset,model) points, "
          f"overall Spearman={analysis['spearman_overall']}")
    for model, s in analysis["per_model"].items():
        print(f"  {model}: n={s['n']} mean_gap={s['mean_gap']} "
              f"mean_sdg_effect={s['mean_sdg_effect']} spearman={s['spearman']}")


if __name__ == "__main__":
    main()
