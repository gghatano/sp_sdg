"""Subject-count reduction analysis (Phase 5).

From subject-count learning curves, estimate N* = the number of real subjects
needed to reach the pre-registered target metric, per augmentation. Then:
  reduction_rate = 1 - N*(aug) / N*(none)
  equivalent_subjects_saved = N*(none) - N*(aug)
with bootstrap confidence intervals over the per-count repeats.

N* is found by linear interpolation between the two consecutive subject counts
whose mean metric brackets the target (monotonicity is not assumed; the first
upward crossing is used). If the target is never reached, N* is None.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

import numpy as np


def _curve_points(rows: list[dict], augmentation: str, metric: str) -> dict[int, list[float]]:
    """subject_count -> list of per-repeat metric values for one augmentation."""
    pts: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("augmentation") != augmentation or r.get("status") != "completed":
            continue
        if metric not in r or "subject_count" not in r:
            continue
        pts[int(r["subject_count"])].append(float(r[metric]))
    return pts


def interpolate_nstar(count_to_value: dict[int, float], target: float) -> float | None:
    """First subject count (interpolated) at which the curve reaches target."""
    counts = sorted(count_to_value)
    if not counts:
        return None
    if count_to_value[counts[0]] >= target:
        return float(counts[0])
    for lo, hi in zip(counts, counts[1:]):
        v_lo, v_hi = count_to_value[lo], count_to_value[hi]
        if v_lo < target <= v_hi:
            frac = (target - v_lo) / (v_hi - v_lo) if v_hi != v_lo else 0.0
            return lo + frac * (hi - lo)
    return None


def _bootstrap_nstar_ci(
    points: dict[int, list[float]], target: float, n_boot: int = 2000, seed: int = 0
) -> tuple[float | None, float | None]:
    counts = sorted(points)
    if not counts or min(len(points[c]) for c in counts) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        resampled = {c: float(np.mean(rng.choice(points[c], size=len(points[c]), replace=True))) for c in counts}
        ns = interpolate_nstar(resampled, target)
        if ns is not None:
            estimates.append(ns)
    if len(estimates) < n_boot * 0.5:  # target unreached in most resamples
        return None, None
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def reduction_analysis(
    rows: list[dict], target: float, metric: str = "accuracy", augmentations: list[str] | None = None, seed: int = 0
) -> dict:
    """Full reduction table vs the none baseline.

    Subject-count rows are identified by carrying a subject_count (not by a fixed
    phase number), so both UCI HAR (phase 4) and later subject datasets such as
    WISDM (phase 6, issue #21) reuse this analysis unchanged.
    """
    subject_rows = [r for r in rows if r.get("subject_count") is not None]
    if augmentations is None:
        augmentations = sorted({r["augmentation"] for r in subject_rows})

    none_points = _curve_points(subject_rows, "none", metric)
    none_means = {c: statistics.mean(v) for c, v in none_points.items()}
    n_star_none = interpolate_nstar(none_means, target)
    none_ci = _bootstrap_nstar_ci(none_points, target, seed=seed)

    results = []
    for aug in augmentations:
        pts = _curve_points(subject_rows, aug, metric)
        if not pts:
            continue
        means = {c: statistics.mean(v) for c, v in pts.items()}
        n_star = interpolate_nstar(means, target)
        ci = _bootstrap_nstar_ci(pts, target, seed=seed)
        reduction = None
        saved = None
        if n_star is not None and n_star_none is not None and n_star_none > 0:
            reduction = round(1 - n_star / n_star_none, 3)
            saved = round(n_star_none - n_star, 2)
        results.append(
            {
                "augmentation": aug,
                "n_star": round(n_star, 2) if n_star is not None else None,
                "n_star_ci": [round(c, 2) if c is not None else None for c in ci],
                "reduction_rate": reduction,
                "subjects_saved": saved,
                "curve": [{"subject_count": c, "mean": round(means[c], 4)} for c in sorted(means)],
            }
        )
    return {
        "target_metric": metric,
        "target_value": target,
        "n_star_none": round(n_star_none, 2) if n_star_none is not None else None,
        "n_star_none_ci": [round(c, 2) if c is not None else None for c in none_ci],
        "methods": results,
    }
