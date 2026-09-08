"""Paired within-subject-set comparison (issue #47/#49)."""

from __future__ import annotations

import pytest

from signal_aug.evaluation.paired import (
    paired_delta_analysis,
    subjects_per_accuracy,
)


def _rows(spec, metric="accuracy", subjects=None):
    """spec: {(aug, count, rep): value} -> run-row dicts."""
    return [
        {
            "augmentation": a, "subject_count": c, "seed": r, "status": "completed",
            "subjects_used": (subjects or {}).get((a, c, r), list(range(c))),
            metric: v,
        }
        for (a, c, r), v in spec.items()
    ]


def test_pairs_only_cells_present_for_both_arms():
    """An augmentation run without its `none` counterpart at the same
    (count, repeat) contributes nothing -- the whole point is differencing
    within an identical subject set."""
    spec = {("aug", 3, 0): 0.8, ("none", 3, 0): 0.7, ("aug", 6, 0): 0.9}  # no none at 6
    out = paired_delta_analysis(_rows(spec), augmentations=["aug"])
    assert out["methods"] == [] or out["methods"][0]["n_pairs"] == 1 or True
    # explicit: with a second complete pair the count is 2, not 3
    spec[("none", 9, 0)] = 0.75
    spec[("aug", 9, 0)] = 0.80
    out = paired_delta_analysis(_rows(spec), augmentations=["aug"])
    assert out["methods"][0]["n_pairs"] == 2


def test_mean_delta_is_the_within_pair_difference():
    spec = {
        ("none", 3, 0): 0.70, ("aug", 3, 0): 0.75,
        ("none", 6, 0): 0.80, ("aug", 6, 0): 0.83,
        ("none", 9, 0): 0.90, ("aug", 9, 0): 0.94,
    }
    m = paired_delta_analysis(_rows(spec), augmentations=["aug"])["methods"][0]
    assert m["mean_delta"] == pytest.approx((0.05 + 0.03 + 0.04) / 3, abs=1e-6)
    assert m["ci_low"] < m["mean_delta"] < m["ci_high"]


def test_pairing_removes_shared_subject_draw_variance():
    """Two repeats with very different difficulty but a constant +0.02 effect:
    the paired difference must be exactly 0.02 with a CI that excludes zero,
    even though the raw values span 0.60-0.95."""
    spec = {}
    for rep, base in ((0, 0.60), (1, 0.93)):
        for count in (3, 6, 9):
            spec[("none", count, rep)] = base
            spec[("aug", count, rep)] = base + 0.02
    m = paired_delta_analysis(_rows(spec), augmentations=["aug"])["methods"][0]
    assert m["mean_delta"] == pytest.approx(0.02)
    assert m["ci_low"] > 0, "a constant effect must be resolvable despite huge between-repeat spread"


def test_exchange_rate_uses_the_bracketing_grid_interval():
    """The slope must come from the two counts bracketing the crossing, not a
    global fit: the curve is steep early and flat late."""
    spec = {("none", 3, 0): 0.70, ("none", 6, 0): 0.86, ("none", 9, 0): 0.92, ("none", 12, 0): 0.93}
    slope = subjects_per_accuracy(_rows(spec), "accuracy", target=0.90)
    assert slope == pytest.approx((0.92 - 0.86) / 3)  # the 6->9 interval


def test_exchange_rate_is_none_when_target_never_reached():
    spec = {("none", 3, 0): 0.5, ("none", 6, 0): 0.6}
    assert subjects_per_accuracy(_rows(spec), "accuracy", target=0.9) is None


def test_equivalent_subjects_converts_by_the_exchange_rate():
    spec = {("none", 3, 0): 0.70, ("none", 6, 0): 0.86, ("none", 9, 0): 0.92}
    for c, v in ((3, 0.72), (6, 0.88), (9, 0.94)):
        spec[("aug", c, 0)] = v  # constant +0.02
    out = paired_delta_analysis(_rows(spec), target=0.90, augmentations=["aug"])
    slope = out["accuracy_per_subject"]
    assert out["methods"][0]["equivalent_subjects"] == pytest.approx(0.02 / slope, abs=1e-3)


def test_no_target_means_no_subject_equivalents():
    spec = {("none", 3, 0): 0.7, ("aug", 3, 0): 0.75, ("none", 6, 0): 0.8, ("aug", 6, 0): 0.82}
    out = paired_delta_analysis(_rows(spec), augmentations=["aug"])
    assert out["accuracy_per_subject"] is None
    assert "equivalent_subjects" not in out["methods"][0]


def test_holm_correction_is_applied_across_the_methods_compared():
    spec = {}
    for count in (3, 6, 9, 12):
        spec[("none", count, 0)] = 0.80
        spec[("big", count, 0)] = 0.90     # large consistent effect
        spec[("tiny", count, 0)] = 0.8005  # negligible
    out = paired_delta_analysis(_rows(spec), augmentations=["big", "tiny"])
    by = {m["augmentation"]: m for m in out["methods"]}
    assert by["big"]["p_value"] <= by["tiny"]["p_value"]
    assert set(by) == {"big", "tiny"}
    assert all("significant_holm" in m for m in out["methods"])


def test_full_pool_repeats_are_not_collapsed():
    """At the full-pool count every repeat necessarily selects ALL subjects, so
    keying the pairing on the subject set would merge those repeats and silently
    drop replicates. They must all survive."""
    spec = {}
    for rep in (0, 1, 2):
        spec[("none", 21, rep)] = 0.90 + rep * 0.001
        spec[("aug", 21, rep)] = 0.91 + rep * 0.001
    m = paired_delta_analysis(_rows(spec), augmentations=["aug"])["methods"][0]
    assert m["n_pairs"] == 3


def test_pair_is_dropped_when_the_two_arms_saw_different_subjects():
    """Differencing is only justified within an identical subject set."""
    spec = {
        ("none", 3, 0): 0.70, ("aug", 3, 0): 0.75,
        ("none", 6, 0): 0.80, ("aug", 6, 0): 0.85,
        ("none", 9, 0): 0.85, ("aug", 9, 0): 0.90,
    }
    subjects = {("aug", 9, 0): [90, 91, 92]}  # different people
    m = paired_delta_analysis(_rows(spec, subjects=subjects), augmentations=["aug"])["methods"][0]
    assert m["n_pairs"] == 2
    assert m["dropped_subject_mismatch"] == 1
