"""Statistical comparison of augmentations against the no-augmentation
baseline (Phase 2, spec section 5).

Wilcoxon signed-rank test over paired per-run metric values, pairing each
augmented run with the baseline run of identical (dataset, model, seed,
train_fraction). Only completed runs participate.
"""

from __future__ import annotations

import math

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
            if all(d == 0 for d in deltas):
                # Wilcoxon is undefined when every paired difference is zero.
                # Depending on the SciPy version this raises or returns NaN;
                # normalize both to None so callers see a single "undefined".
                p_value = None
            else:
                try:
                    _, p_value = wilcoxon(deltas)
                    p_value = float(p_value)
                    if math.isnan(p_value):
                        p_value = None
                except ValueError:
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


def holm_bonferroni(entries: list[dict], alpha: float = 0.05, p_key: str = "p_value") -> list[dict]:
    """Annotate each entry with `significant_holm` using the Holm-Bonferroni
    step-down procedure over the family of tests with a non-None p-value.

    Entries are returned sorted by p-value ascending, each gaining a
    `p_adjusted` field. Correct step-down semantics: the adjusted p-values are
    made monotone non-decreasing (cumulative max), so once a hypothesis fails to
    be rejected, no later (larger-p) hypothesis can be rejected. Entries with a
    None p-value are returned unchanged (not significant) after the tested ones.
    """
    testable = sorted([e for e in entries if e.get(p_key) is not None], key=lambda e: e[p_key])
    untestable = [e for e in entries if e.get(p_key) is None]
    m = len(testable)
    running_max = 0.0
    for i, e in enumerate(testable):
        adj = min(1.0, e[p_key] * (m - i))
        running_max = max(running_max, adj)  # enforce monotone non-decreasing
        e["p_adjusted"] = running_max
        e["significant_holm"] = bool(running_max < alpha)
    for e in untestable:
        e["p_adjusted"] = None
        e["significant_holm"] = False
    return testable + untestable
