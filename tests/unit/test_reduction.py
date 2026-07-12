"""N* interpolation and reduction-rate analysis (Phase 5)."""

from signal_aug.evaluation.reduction import interpolate_nstar, reduction_analysis


def test_interpolate_basic_crossing():
    # target 0.90 between n=6 (0.88) and n=9 (0.92) -> 6 + 0.5*3 = 7.5
    curve = {3: 0.80, 6: 0.88, 9: 0.92}
    assert interpolate_nstar(curve, 0.90) == 7.5


def test_interpolate_already_reached_at_first():
    assert interpolate_nstar({3: 0.95, 6: 0.96}, 0.90) == 3.0


def test_interpolate_never_reached():
    assert interpolate_nstar({3: 0.5, 6: 0.6}, 0.90) is None


def test_interpolate_uses_first_upward_crossing():
    # non-monotone curve; first crossing at 6->9
    curve = {3: 0.80, 6: 0.85, 9: 0.91, 12: 0.88, 15: 0.93}
    ns = interpolate_nstar(curve, 0.90)
    assert 6 < ns <= 9


def _rows(curve_by_aug):
    rows = []
    for aug, curve in curve_by_aug.items():
        for count, vals in curve.items():
            for v in vals:
                rows.append({"phase": 4, "augmentation": aug, "subject_count": count,
                             "status": "completed", "accuracy": v})
    return rows


def test_reduction_positive_when_aug_curve_is_higher():
    rows = _rows({
        "none": {3: [0.80], 6: [0.85], 9: [0.91]},   # N* ~ 8.5
        "mixup": {3: [0.86], 6: [0.91], 9: [0.94]},  # N* ~ 5.5
    })
    res = reduction_analysis(rows, target=0.90, metric="accuracy")
    mixup = next(m for m in res["methods"] if m["augmentation"] == "mixup")
    assert mixup["n_star"] < res["n_star_none"]
    assert mixup["reduction_rate"] > 0
    assert mixup["subjects_saved"] > 0


def test_reduction_none_is_zero():
    rows = _rows({"none": {3: [0.80], 6: [0.88], 9: [0.92]}})
    res = reduction_analysis(rows, target=0.90, metric="accuracy")
    none = next(m for m in res["methods"] if m["augmentation"] == "none")
    assert none["reduction_rate"] == 0.0


def test_reduction_handles_unreached_target():
    rows = _rows({"none": {3: [0.5], 6: [0.6]}, "mixup": {3: [0.55], 6: [0.65]}})
    res = reduction_analysis(rows, target=0.90, metric="accuracy")
    assert res["n_star_none"] is None
    mixup = next(m for m in res["methods"] if m["augmentation"] == "mixup")
    assert mixup["reduction_rate"] is None
