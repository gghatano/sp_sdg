"""H2 (issue #12): correlate the SDG effect magnitude with the no-augmentation
generalization gap.

For each (dataset, model):
  gap        = mean_seed[ train_accuracy(none) - test_accuracy(none) ]
  sdg_effect = mean over augmentations of the mean-seed accuracy delta vs none
               (i.e. how much augmentation helps, on average, for that pair)

H2 predicts a positive correlation: models/datasets that overfit more (large
gap) benefit more from augmentation. We report Spearman rank correlation, which
is robust to the small, non-normal sample of (dataset, model) points.
"""

from __future__ import annotations

import statistics
from collections import defaultdict


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def generalization_gap(none_rows: list[dict]) -> dict[tuple, float]:
    """(dataset, model) -> mean(train_accuracy - accuracy) over seeds, for none runs."""
    by_pair = defaultdict(list)
    for r in none_rows:
        if r.get("status") != "completed" or r.get("augmentation") != "none":
            continue
        if "train_accuracy" not in r or "accuracy" not in r:
            continue
        by_pair[(r["dataset"], r["model"])].append(r["train_accuracy"] - r["accuracy"])
    return {k: statistics.mean(v) for k, v in by_pair.items() if v}


def sdg_effect(summary: list[dict], train_fraction: float = 1.0) -> dict[tuple, float]:
    """(dataset, model) -> mean over augmentations of (accuracy_mean - none accuracy_mean)
    at the given train_fraction, from the aggregated summary."""
    def frac(s):
        return s.get("train_fraction", 1.0)

    baseline = {
        (s["dataset"], s["model"]): s["accuracy_mean"]
        for s in summary
        if s["augmentation"] == "none" and frac(s) == train_fraction
    }
    deltas = defaultdict(list)
    for s in summary:
        if s["augmentation"] == "none" or frac(s) != train_fraction:
            continue
        base = baseline.get((s["dataset"], s["model"]))
        if base is not None:
            deltas[(s["dataset"], s["model"])].append(s["accuracy_mean"] - base)
    return {k: statistics.mean(v) for k, v in deltas.items() if v}


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(xs, ys)
        return None if rho != rho else float(rho)  # NaN guard
    except Exception:
        return None


def analyze_gap(none_rows: list[dict], summary: list[dict], train_fraction: float = 1.0) -> dict:
    gaps = generalization_gap(none_rows)
    effects = sdg_effect(summary, train_fraction=train_fraction)
    pairs = sorted(set(gaps) & set(effects))
    points = [
        {"dataset": d, "model": m, "gap": round(gaps[(d, m)], 4), "sdg_effect": round(effects[(d, m)], 4)}
        for (d, m) in pairs
    ]
    xs = [p["gap"] for p in points]
    ys = [p["sdg_effect"] for p in points]
    overall = _spearman(xs, ys)
    per_model = {}
    for model in sorted({m for _, m in pairs}):
        mx = [p["gap"] for p in points if p["model"] == model]
        my = [p["sdg_effect"] for p in points if p["model"] == model]
        per_model[model] = {
            "n": len(mx),
            "mean_gap": round(_mean(mx), 4) if mx else None,
            "mean_sdg_effect": round(_mean(my), 4) if my else None,
            "spearman": _spearman(mx, my),
        }
    return {
        "train_fraction": train_fraction,
        "n_points": len(points),
        "spearman_overall": overall,
        "per_model": per_model,
        "points": points,
    }
