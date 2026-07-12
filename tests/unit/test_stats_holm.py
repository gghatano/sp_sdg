"""Holm-Bonferroni step-down correction (stats.holm_bonferroni)."""

from signal_aug.evaluation.stats import holm_bonferroni


def _entries(pvals):
    return [{"augmentation": f"a{i}", "p_value": p} for i, p in enumerate(pvals)]


def test_smallest_p_uses_full_family_size():
    # m=4; smallest p * 4 must clear alpha to be significant
    out = holm_bonferroni(_entries([0.01, 0.2, 0.3, 0.4]), alpha=0.05)
    first = out[0]
    assert first["p_value"] == 0.01
    assert first["p_adjusted"] == 0.04  # 0.01 * 4
    assert first["significant_holm"] is True


def test_step_down_monotonicity_no_later_rejection_after_failure():
    # After the first non-rejection, no larger-p hypothesis may be rejected,
    # even if its own p*(m-i) would fall under alpha.
    out = holm_bonferroni(_entries([0.04, 0.049, 0.001]), alpha=0.05)
    # sorted: 0.001, 0.04, 0.049
    by_p = {e["p_value"]: e for e in out}
    assert by_p[0.001]["significant_holm"] is True   # 0.001*3=0.003
    assert by_p[0.04]["significant_holm"] is False    # 0.04*2=0.08 -> fail
    # 0.049*1=0.049 < 0.05 in isolation, but monotonicity forbids rejecting it
    assert by_p[0.049]["significant_holm"] is False
    assert by_p[0.049]["p_adjusted"] >= by_p[0.04]["p_adjusted"]  # non-decreasing


def test_p_adjusted_is_non_decreasing():
    out = holm_bonferroni(_entries([0.001, 0.01, 0.02, 0.9]), alpha=0.05)
    adj = [e["p_adjusted"] for e in out]
    assert adj == sorted(adj)  # monotone non-decreasing


def test_single_hypothesis_family():
    out = holm_bonferroni(_entries([0.049]), alpha=0.05)
    assert out[0]["p_adjusted"] == 0.049
    assert out[0]["significant_holm"] is True


def test_none_pvalues_excluded_and_not_significant():
    entries = _entries([0.001]) + [{"augmentation": "x", "p_value": None}]
    out = holm_bonferroni(entries, alpha=0.05)
    x = next(e for e in out if e["augmentation"] == "x")
    assert x["significant_holm"] is False
    assert x["p_adjusted"] is None
    # family size counts only testable entries: 0.001 * 1 significant
    a0 = next(e for e in out if e["p_value"] == 0.001)
    assert a0["significant_holm"] is True


def test_order_invariance():
    a = holm_bonferroni(_entries([0.3, 0.01, 0.2]), alpha=0.05)
    b = holm_bonferroni(_entries([0.01, 0.2, 0.3]), alpha=0.05)
    sig_a = {e["p_value"]: e["significant_holm"] for e in a}
    sig_b = {e["p_value"]: e["significant_holm"] for e in b}
    assert sig_a == sig_b


def test_adjusted_p_capped_at_one():
    out = holm_bonferroni(_entries([0.5, 0.6]), alpha=0.05)
    assert all(e["p_adjusted"] <= 1.0 for e in out)
