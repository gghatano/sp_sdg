"""Statistical comparison of augmentations against the no-augmentation
baseline (Phase 2, spec section 5).

Wilcoxon signed-rank test over paired per-run metric values, pairing each
augmented run with the baseline run of identical (dataset, model, seed,
train_fraction). Only completed runs participate.
"""

from __future__ import annotations

from scipy.stats import wilcoxon


def paired_deltas(rows: list[dict], augmentation: str, model: str, metric: str = "accuracy") -> list[float]:
    """Per-pair metric differences (augmented - baseline) across datasets/seeds/fractions."""
    def key(r: dict) -> tuple:
        return (r["dataset"], r["model"], r["seed"], r.get("train_fraction", 1.0))

    baseline = {
        key(r): r[metric]
        for r in rows
        if r["status"] == "completed" and r["augmentation"] == "none" and metric in r
    }
    deltas = []
    for r in rows:
        if r["status"] != "completed" or r["augmentation"] != augmentation or metric not in r:
            continue
        if r["model"] != model:
            continue
        base = baseline.get(key(r))
        if base is not None:
            deltas.append(r[metric] - base)
    return deltas


def wilcoxon_vs_none(rows: list[dict], metric: str = "accuracy") -> list[dict]:
    """Wilcoxon signed-rank test per (augmentation, model) vs the none baseline.

    Returns entries with n pairs, mean delta, and p-value (None when the test
    is undefined, e.g. all-zero differences or too few pairs).
    """
    augmentations = sorted({r["augmentation"] for r in rows if r["augmentation"] != "none"})
    models = sorted({r["model"] for r in rows})
    results = []
    for aug in augmentations:
        for model in models:
            deltas = paired_deltas(rows, aug, model, metric)
            if len(deltas) < 5:
                continue
            mean_delta = sum(deltas) / len(deltas)
            try:
                stat, p_value = wilcoxon(deltas)
                p_value = float(p_value)
            except ValueError:  # e.g. all differences are zero
                p_value = None
            results.append(
                {
                    "augmentation": aug,
                    "model": model,
                    "metric": metric,
                    "n_pairs": len(deltas),
                    "mean_delta": round(mean_delta, 4),
                    "p_value": p_value,
                }
            )
    return results
