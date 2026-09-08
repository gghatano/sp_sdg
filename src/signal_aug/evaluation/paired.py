"""Paired within-subject-set comparison for the subject-count experiments.

Why this exists (issue #47/#49). The pre-registered estimand was N*(aug)/N*(none)
-- the ratio of subject counts needed to hit a target. Null calibration showed
that ratio cannot resolve the effects actually present: comparing `none` against
itself with the same 3 repeats yields apparent reductions up to 38% (UCI HAR) /
22% (WISDM) at the one-sided 95th percentile, while every observed effect is
around 10%.

The subject-count grid, however, is already a PAIRED design and the N* route
throws that away. At a given (subject_count, repeat) every augmentation is
trained on the identical subject set -- verified across all 126 UCI HAR runs.
Differencing within a pair removes the subject-draw variance, which dominates:
at N=3 the accuracy spread across repeats is sd 0.038, far larger than any
augmentation effect.

The paired difference is sensitive enough to separate the negative control from
the real methods on both datasets, which the N* route could not do.

Deviation notice: making this the primary estimand is a post-hoc change, decided
after seeing the null calibration. It is recorded in artifacts/deviations.md and
J-PAIRED-PRIMARY. The pre-registered N* analysis is retained and reported
alongside, unchanged.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from scipy.stats import wilcoxon


def _pairs(
    rows: list[dict], augmentation: str, metric: str
) -> tuple[list[tuple[int, int, float, float]], int]:
    """(subject_count, repeat, value_aug, value_none) for cells present in both,
    plus the number of cells dropped because the two arms did not train on the
    same subjects.

    Keyed on (subject_count, repeat), NOT on the subject set itself: at the
    full-pool count every repeat necessarily selects all subjects, so keying on
    the set would collapse those repeats into one and silently discard them.
    The subject set is instead used to VERIFY the pairing -- differencing two
    runs is only justified when they saw the same subjects, so a mismatch drops
    the pair rather than quietly biasing the difference.
    """
    by: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r.get("status") != "completed" or r.get("subject_count") is None or metric not in r:
            continue
        if r.get("seed") is None:
            continue
        by[(int(r["subject_count"]), int(r["seed"]))][r["augmentation"]] = r

    out, mismatched = [], 0
    for (count, rep), cell in sorted(by.items()):
        if augmentation not in cell or "none" not in cell:
            continue
        a, n = cell[augmentation], cell["none"]
        sa, sn = a.get("subjects_used"), n.get("subjects_used")
        if sa is not None and sn is not None and sorted(sa) != sorted(sn):
            mismatched += 1
            continue
        out.append((count, rep, float(a[metric]), float(n[metric])))
    return out, mismatched


def subjects_per_accuracy(rows: list[dict], metric: str, target: float) -> float | None:
    """Local slope of the `none` curve near the target crossing, in accuracy per
    subject, used to express a paired accuracy difference in subject-equivalents.

    Taken from the two grid points that bracket the crossing rather than a global
    fit: the curve is steep at small N and flat at large N, so a single slope
    would misstate the exchange rate exactly where it matters.
    """
    curve: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("augmentation") != "none" or r.get("status") != "completed":
            continue
        if r.get("subject_count") is None or metric not in r:
            continue
        curve[int(r["subject_count"])].append(float(r[metric]))
    means = {c: statistics.mean(v) for c, v in curve.items()}
    counts = sorted(means)
    for lo, hi in zip(counts, counts[1:]):
        if means[lo] < target <= means[hi]:
            rise = means[hi] - means[lo]
            return rise / (hi - lo) if hi != lo else None
    return None


def paired_delta_analysis(
    rows: list[dict],
    metric: str = "accuracy",
    target: float | None = None,
    augmentations: list[str] | None = None,
) -> dict:
    """Paired difference vs `none` within identical subject sets.

    Returns per-augmentation mean difference, normal-approximation 95% CI,
    Wilcoxon signed-rank p, Holm-corrected significance, and -- when `target` is
    given and the `none` curve crosses it -- the difference expressed in
    equivalent subjects.
    """
    subject_rows = [r for r in rows if r.get("subject_count") is not None]
    if augmentations is None:
        augmentations = sorted({r["augmentation"] for r in subject_rows} - {"none"})

    slope = subjects_per_accuracy(subject_rows, metric, target) if target is not None else None

    methods = []
    for aug in augmentations:
        pairs, mismatched = _pairs(subject_rows, aug, metric)
        if len(pairs) < 2:
            continue
        deltas = [a - n for _, _, a, n in pairs]
        mean = statistics.mean(deltas)
        sd = statistics.stdev(deltas)
        se = sd / math.sqrt(len(deltas))
        entry = {
            "augmentation": aug,
            "n_pairs": len(deltas),
            "mean_delta": round(mean, 5),
            "sd": round(sd, 5),
            "ci_low": round(mean - 1.96 * se, 5),
            "ci_high": round(mean + 1.96 * se, 5),
            "p_value": round(float(wilcoxon(deltas).pvalue), 5) if any(d != 0 for d in deltas) else 1.0,
            "dropped_subject_mismatch": mismatched,
        }
        if slope:
            # positive = augmentation is worth this many extra subjects
            entry["equivalent_subjects"] = round(mean / slope, 3)
            entry["equivalent_subjects_ci"] = [
                round((mean - 1.96 * se) / slope, 3),
                round((mean + 1.96 * se) / slope, 3),
            ]
        methods.append(entry)

    # Holm across the methods compared in this table
    for rank, m in enumerate(sorted(methods, key=lambda x: x["p_value"])):
        m["significant_holm"] = m["p_value"] < 0.05 / (len(methods) - rank)

    return {
        "metric": metric,
        "target": target,
        "accuracy_per_subject": round(slope, 6) if slope else None,
        "methods": sorted(methods, key=lambda m: m["augmentation"]),
    }
