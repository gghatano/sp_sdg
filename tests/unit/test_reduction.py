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


# --- interpolate_nstar boundary / non-monotone cases (#4-15) ---

def test_interpolate_exact_target_at_upper_point():
    # v_hi == target exactly: crossing resolves at the upper count
    curve = {3: 0.85, 6: 0.90}
    assert interpolate_nstar(curve, 0.90) == 6.0


def test_interpolate_flat_segment_at_target():
    # v_lo == v_hi (both below then jump): no division by zero
    curve = {3: 0.80, 6: 0.80, 9: 0.95}
    ns = interpolate_nstar(curve, 0.90)
    assert 6 < ns <= 9


def test_interpolate_first_upward_crossing_wins_over_later_dip():
    # non-monotone: crosses at 6->9, dips at 12, recovers at 15
    curve = {3: 0.80, 6: 0.85, 9: 0.91, 12: 0.88, 15: 0.93}
    ns = interpolate_nstar(curve, 0.90)
    assert 6 < ns <= 9


def test_interpolate_empty_curve_returns_none():
    assert interpolate_nstar({}, 0.90) is None


# --- bootstrap CI (#4-4): needs multiple repeats with real variance ---

def _rows_multi(curve_by_aug):
    rows = []
    for aug, curve in curve_by_aug.items():
        for count, vals in curve.items():
            for v in vals:
                rows.append({"phase": 4, "augmentation": aug, "subject_count": count,
                             "status": "completed", "accuracy": v})
    return rows


def test_bootstrap_ci_brackets_point_estimate():
    # 4 repeats per count with spread around the crossing
    rows = _rows_multi({"none": {
        3: [0.80, 0.82, 0.79, 0.81],
        6: [0.88, 0.90, 0.87, 0.89],
        9: [0.93, 0.94, 0.92, 0.95],
    }})
    res = reduction_analysis(rows, target=0.90, metric="accuracy", seed=0)
    none = next(m for m in res["methods"] if m["augmentation"] == "none")
    lo, hi = none["n_star_ci"]
    assert lo is not None and hi is not None
    assert lo <= res["n_star_none"] <= hi
    assert 3.0 <= lo <= hi <= 9.0


def test_bootstrap_ci_deterministic_for_seed():
    rows = _rows_multi({"none": {3: [0.80, 0.83], 6: [0.88, 0.91], 9: [0.93, 0.95]}})
    a = reduction_analysis(rows, target=0.90, seed=7)["n_star_none_ci"]
    b = reduction_analysis(rows, target=0.90, seed=7)["n_star_none_ci"]
    assert a == b


def test_bootstrap_ci_none_when_target_mostly_unreached():
    # curve peaks well below target: N* undefined in most resamples -> CI None
    rows = _rows_multi({"none": {3: [0.5, 0.52], 6: [0.6, 0.61], 9: [0.7, 0.69]}})
    res = reduction_analysis(rows, target=0.90, seed=0)
    assert res["n_star_none"] is None
    assert res["n_star_none_ci"] == [None, None]
