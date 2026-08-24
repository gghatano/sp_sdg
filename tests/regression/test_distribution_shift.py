"""Regression: the 学習と評価 tab must show, per dataset, how each augmentation
changed the training distribution — computed by running the production
augmenters against that dataset's real data (0826 discussion prep, requested
alongside the paper→report rename and dashboard removal)."""

import json
from pathlib import Path

from signal_aug.augmentations.methods import REGISTRY
from signal_aug.reporting.build import gather_context, render_report

# datasets with local training data (same set as dataset_samples.json)
WITH_LOCAL_DATA = {
    "ECG5000", "FordA", "GunPoint", "ECG200", "TwoLeadECG", "ItalyPowerDemand",
    "MoteStrain", "SonyAIBORobotSurface1", "CBF", "Wafer",
    "ArrowHead", "Coffee", "Plane",  # appendix (non-temporal)
    "UCI_HAR", "WISDM",
}
NO_LOCAL_DATA = {"PAMAP2", "WESAD"}


def _shift_file() -> dict:
    return json.loads(Path("report/assets/data/distribution_shift.json").read_text(encoding="utf-8"))


def test_every_dataset_with_local_data_has_a_shift_entry():
    shift = _shift_file()
    assert set(shift) == WITH_LOCAL_DATA, (
        f"distribution_shift.json covers {sorted(shift)}, expected {sorted(WITH_LOCAL_DATA)}"
    )


def test_shift_entries_cover_every_non_baseline_method():
    shift = _shift_file()
    expected = set(REGISTRY) - {"none"}
    for key, entry in shift.items():
        assert set(entry["methods"]) == expected, f"{key}: methods {sorted(entry['methods'])} != {sorted(expected)}"


def test_shift_metrics_are_bounded_and_synthetic_counts_are_positive():
    shift = _shift_file()
    for key, entry in shift.items():
        for name, m in entry["methods"].items():
            assert 0.0 <= m["class_balance_shift"] <= 1.0, f"{key}/{name}: TV distance out of [0,1]"
            assert m["n_synthetic"] > 0, f"{key}/{name}: produced no synthetic samples"
            if m["std_ratio"] is not None:
                assert m["std_ratio"] > 0


def test_evaluation_rows_carry_distribution_or_an_unavailable_note():
    context = gather_context(".")
    rows = {r["key"]: r for r in context["eval_tab"]["all_rows"]}
    for key in WITH_LOCAL_DATA:
        assert rows[key]["distribution"], f"{key}: missing distribution rows"
    for key in NO_LOCAL_DATA:
        if key in rows:
            assert rows[key]["distribution"] is None, f"{key}: should have no computed distribution"


def test_distribution_section_renders_in_evaluation_and_appendix_tabs():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    assert "拡張によるデータ分布の変化を見る" in html
    # unavailable note only where there is no local data (subject datasets w/o samples)
    assert "分布変化は未算出" in html


def test_distribution_table_uses_real_computed_numbers_not_placeholders():
    """A row's class-balance shift must match the committed JSON exactly (no
    rounding drift, no hand-typed stand-in values)."""
    shift = _shift_file()
    context = gather_context(".")
    rows = {r["key"]: r for r in context["eval_tab"]["all_rows"]}
    for key in ("ECG5000", "UCI_HAR"):
        by_method = {m["augmentation"]: m for m in rows[key]["distribution"]}
        for name, expected in shift[key]["methods"].items():
            assert by_method[name]["class_balance_shift"] == expected["class_balance_shift"]
            assert by_method[name]["std_ratio"] == expected["std_ratio"]


def test_mechanism_note_matches_the_real_implementation():
    """The explanatory note claims mixup/dtw/label_shuffle draw synthetic
    samples roughly uniformly across classes (not proportional to prevalence),
    which is a specific, checkable property of the augmenters — verify it
    against a synthetic imbalanced example rather than trusting the prose."""
    import numpy as np

    rng = np.random.default_rng(0)
    # 3 classes, heavily imbalanced: 100 / 5 / 5
    y = np.array([0] * 100 + [1] * 5 + [2] * 5)
    X = rng.normal(size=(len(y), 1, 8)).astype(np.float32)

    from signal_aug.augmentations.methods import apply_augmentation

    for method in ("mixup", "dtw"):
        _, y_aug = apply_augmentation(method, X, y, seed=0, params={"ratio": 1.0})
        synth_labels = y_aug[len(y):]
        # uniform-over-classes sampling should give each class roughly 1/3 of
        # the synthetic portion, far from the original 100:5:5 prevalence
        counts = np.bincount(synth_labels, minlength=3)
        assert counts[1] / len(synth_labels) > 0.15, f"{method}: minority class underrepresented in synthetic portion"

    _, y_over = apply_augmentation("oversample", X, y, seed=0, params={"ratio": 1.0})
    synth_over = y_over[len(y):]
    counts_over = np.bincount(synth_over, minlength=3)
    # proportional sampling should track the original imbalance closely
    assert counts_over[1] / len(synth_over) < 0.15, "oversample: unexpectedly rebalanced classes"
