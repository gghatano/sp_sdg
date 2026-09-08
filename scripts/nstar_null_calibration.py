"""Null calibration of the N* estimator (issue #47).

The headline analysis judges a reduction against label_shuffle. That control is
a TREATMENT acting on training, not a zero-effect condition, so its apparent
reduction is not "what zero effect looks like". This script measures that
directly: it takes many repeats of `none` alone and compares disjoint sets of
repeats against each other. Two such sets differ by construction only in the
subject draw and the training seed, so any reduction between them is the
estimator's own variability under a true null.

Design points (from external review):

* Sets are TRIPLES, matching the 3 repeats the headline analysis averages.
  Calibrating with 10-repeat means would produce a more stable curve than the
  one being calibrated and would understate the noise.
* Replicate indices are assigned to A/B once and reused at every subject count,
  mirroring the headline analysis, which builds each curve from a fixed set of
  replicate indices across counts.
* The reduction ratio is not symmetric under swapping A and B, so the signed
  distribution and the log ratio are both reported.
* The claim being calibrated is directional ("augmentation REDUCES N*"), so the
  one-sided 95th percentile is the relevant quantity, not the two-sided one on
  absolute values.
* Draws where the target is never reached are counted and reported, never
  dropped silently nor replaced by the grid edge.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from signal_aug.evaluation.reduction import interpolate_nstar
from signal_aug.reporting.aggregate import manifest_path


def load_points(manifests_dir: Path, name: str, metric: str, dataset: str) -> dict[int, dict[int, float]]:
    """{subject_count: {replicate_index: metric}} for completed `none` runs.

    Filtered by dataset, not by the run_id prefix alone: experiment names nest
    ("nstar_null" is a prefix of "nstar_null_wisdm"), so a glob on the name
    would silently pool two datasets into one curve and make the target
    unreachable everywhere.
    """
    pts: dict[int, dict[int, float]] = defaultdict(dict)
    for path in sorted(manifests_dir.glob(f"{name}_*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("status") != "completed" or m.get("augmentation") != "none":
            continue
        if m.get("dataset") != dataset:
            continue
        mp = manifest_path(m.get("metrics_path"))
        if not (mp and mp.exists()):
            continue
        value = json.loads(mp.read_text(encoding="utf-8"))[metric]
        pts[int(m["subject_count"])][int(m["seed"])] = float(value)
    return dict(pts)


def calibrate(points, target, n_draws, group, seed=0):
    counts = sorted(points)
    reps = sorted(set.intersection(*(set(points[c]) for c in counts)))
    rng = np.random.default_rng(seed)

    reductions, log_ratios = [], []
    unreachable = 0
    for _ in range(n_draws):
        picked = rng.choice(reps, size=2 * group, replace=False)
        a_idx, b_idx = picked[:group], picked[group:]
        curve_a = {c: float(np.mean([points[c][r] for r in a_idx])) for c in counts}
        curve_b = {c: float(np.mean([points[c][r] for r in b_idx])) for c in counts}
        n_a, n_b = interpolate_nstar(curve_a, target), interpolate_nstar(curve_b, target)
        if n_a is None or n_b is None or n_a <= 0 or n_b <= 0:
            unreachable += 1
            continue
        reductions.append(1.0 - n_b / n_a)
        log_ratios.append(math.log(n_a / n_b))

    r = np.array(reductions)
    return {
        "n_draws": n_draws,
        "group_size": group,
        "n_repeats_available": len(reps),
        "subject_counts": counts,
        "target": target,
        "unreachable_draws": unreachable,
        "unreachable_fraction": round(unreachable / n_draws, 4),
        "usable_draws": len(reductions),
        "reduction_mean": round(float(r.mean()), 4),
        "reduction_median": round(float(np.median(r)), 4),
        # directional: the claim under test is "augmentation reduces N*"
        "reduction_p95_one_sided": round(float(np.percentile(r, 95)), 4),
        "reduction_p99_one_sided": round(float(np.percentile(r, 99)), 4),
        "reduction_abs_p95": round(float(np.percentile(np.abs(r), 95)), 4),
        "log_ratio_mean": round(float(np.mean(log_ratios)), 4),
        "log_ratio_p95": round(float(np.percentile(log_ratios, 95)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiments/nstar_null_calibration.yaml")
    ap.add_argument("--manifests", default="runs/manifests")
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--group", type=int, default=3, help="repeats per side; match the headline analysis")
    ap.add_argument("--out", default="artifacts/nstar_null_calibration.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    points = load_points(Path(args.manifests), cfg["name"], cfg["target_metric"], cfg["dataset"])
    if not points:
        raise SystemExit("no completed runs found - run the calibration grid first")

    result = calibrate(points, float(cfg["target_value"]), args.draws, args.group)
    result["dataset"] = cfg["dataset"]
    result["metric"] = cfg["target_metric"]
    result["interpretation"] = (
        "同一条件(none)同士を主解析と同じ3反復で比較したときに現れる見かけの削減率の分布。"
        "reduction_p95_one_sided は『効果ゼロでもこの値までは出る』というヌル分布の棄却閾値であって、"
        "検出限界(最小検出可能効果)ではない——検出力の計算は行っていない。"
        "この値を下回る観測された削減は、削減効果を裏づける証拠として不十分だと言えるだけで、"
        "『効果がない』ことや『アーティファクトである』ことの証明ではない。"
        "また 5000 draws は 140 runs の再標本化であって独立な追加実験ではなく、"
        "この pool(21名)/test(9名) と本推定手順に条件づけた値である。"
    )
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
