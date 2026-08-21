"""Regression: datasets whose axis is not time (image contour / spectrum) are
excluded from the main analysis and presented in the appendix tab instead.

Time-domain augmentation assumes a time axis — DTW alignment in particular
assumes within-class variation is temporal stretching — so mixing those datasets
into the main aggregate muddies the question the study asks. The exclusion is
post-hoc, so the report must keep the pre-exclusion numbers visible.
"""

import json
import re
from pathlib import Path

import yaml

from signal_aug.reporting.aggregate import manifest_path
from signal_aug.reporting.build import gather_context, render_report

NON_TEMPORAL = {"ArrowHead", "Plane", "Coffee"}


def _config() -> dict:
    return yaml.safe_load(Path("config/datasets.yaml").read_text(encoding="utf-8"))["datasets"]


def test_non_temporal_datasets_are_flagged_in_config():
    """The flag lives in config (config-driven, CLAUDE.md), not in the report."""
    cfg = _config()
    flagged = {k for k, s in cfg.items() if s.get("temporal") is False}
    assert flagged == NON_TEMPORAL, f"temporal flags drifted: {sorted(flagged)}"
    for key in NON_TEMPORAL:
        assert cfg[key].get("axis"), f"{key}: the actual axis must be stated"


def test_main_statistics_exclude_them():
    """表5 (the RQ1 answer) must be computed without the non-temporal datasets."""
    results = json.loads(Path("report/assets/data/results.json").read_text(encoding="utf-8"))
    assert set(results["non_temporal_datasets"]) == NON_TEMPORAL
    context = gather_context(".")
    main_pairs = {s["n_pairs"] for s in context["stats"]}
    all_pairs = {s["n_pairs"] for s in context["stats_all_datasets"]}
    assert main_pairs and all_pairs
    assert max(main_pairs) < max(all_pairs), "main stats still include every dataset"
    # the excluded datasets must not be counted as study datasets in the headline
    facts = context["facts"]
    assert facts["n_ucr_datasets"] == facts["n_ucr_datasets_all"] - facts["n_non_temporal"]


def test_pre_exclusion_numbers_stay_visible():
    """A post-hoc exclusion must be auditable: the all-datasets version of the
    same test, and any flipped verdicts, are shown in the appendix."""
    context = gather_context(".")
    assert context["stats_comparison"], "no before/after comparison built"
    for row in context["stats_comparison"]:
        assert row["main"] and row["all"]
    html = render_report(context, "report/src", css="")
    appendix = html[html.index('data-tab="appendix"'):]
    assert 'id="appendix-comparison"' in appendix
    assert "post-hoc" in appendix
    # every verdict flip is named rather than quietly absorbed
    for flip in context["stats_flips"]:
        assert flip["augmentation"] in appendix


def test_appendix_tab_presents_the_excluded_datasets_in_full():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    assert 'data-tabbtn="appendix"' in html
    appendix = html[html.index('data-tab="appendix"'):]
    for key in NON_TEMPORAL:
        anchor = f"dataset-{key.lower()}"
        assert anchor in appendix, f"{key}: description missing from the appendix"
        assert f"eval-{key.lower()}" in appendix, f"{key}: results missing from the appendix"
    # ...and no longer in the main dataset/evaluation tabs
    main_tabs = html[html.index('data-tab="datasets"'):html.index('data-tab="appendix"')]
    for key in NON_TEMPORAL:
        assert f'id="dataset-{key.lower()}"' not in main_tabs, f"{key} still in the main dataset tab"
        assert f'id="eval-{key.lower()}"' not in main_tabs, f"{key} still in the main evaluation tab"


def test_findings_state_the_exclusion_effect():
    findings = json.loads(Path("artifacts/findings.json").read_text(encoding="utf-8"))["findings"]
    by_id = {f["id"]: f for f in findings}
    assert "F-16" in by_id, "the effect of the exclusion must be recorded as a finding"
    assert "post-hoc" in by_id["F-16"]["notes"]
    # F-5's numbers must come from the main (temporal-only) analysis
    assert "9 データセット" in by_id["F-5"]["claim"] or "9データセット" in by_id["F-5"]["claim"]


def test_manifest_paths_resolve_regardless_of_recording_os():
    """Runs executed on Windows recorded backslash paths; on POSIX those never
    resolved, so whole datasets silently disappeared from the aggregate."""
    assert manifest_path("runs\\metrics\\x.json") == Path("runs/metrics/x.json")
    assert manifest_path("runs/metrics/x.json") == Path("runs/metrics/x.json")
    assert manifest_path(None) is None
    # the datasets that were being dropped are present in the current aggregate
    results = json.loads(Path("report/assets/data/results.json").read_text(encoding="utf-8"))
    for key in ("reduction_pamap2", "reduction_wesad"):
        assert results.get(key), f"{key} missing from the aggregate"


def test_paper_states_the_dataset_basis():
    """A reader must see that the headline count excludes the appendix data."""
    html = render_report(gather_context("."), "report/src", css="")
    paper = html[html.index('data-tab="paper"'):html.index('data-tab="datasets"')]
    assert re.search(r"軸が時間でない", paper), "the exclusion is not stated in the paper tab"
    assert "#tab-appendix" in paper, "the paper does not link to the appendix"


def test_paper_tab_never_mentions_appendix_datasets():
    """The paper tab discusses time-series data only: no appendix dataset may
    appear in its tables, highlighted conditions, or figures."""
    context = gather_context(".")
    for key in ("summary", "best_improvements", "worst_degradations"):
        named = {row["dataset"] for row in context[key]}
        assert not (named & NON_TEMPORAL), f"{key} still names {sorted(named & NON_TEMPORAL)}"
    panels = {p["dataset"] for p in context["curve_panels"]}
    assert not (panels & NON_TEMPORAL), f"learning-curve figures still show {sorted(panels & NON_TEMPORAL)}"

    html = render_report(context, "report/src", css="")
    paper = html[html.index('data-tab="paper"'):html.index('data-tab="datasets"')]
    for key in NON_TEMPORAL:
        # the paper may link to the appendix, but must not present the data
        assert f">{key}<" not in paper, f"{key} appears in the paper tab"


def test_headline_run_count_matches_the_paper_scope():
    """計 N run in the abstract counts the runs the paper actually reports."""
    context = gather_context(".")
    facts = context["facts"]
    results_runs = [r for r in context["results"]["runs"]
                    if r["status"] == "completed" and r["dataset"] != "synthetic"]
    appendix = [r for r in results_runs if r["dataset"] in NON_TEMPORAL]
    assert facts["n_appendix_runs"] == len(appendix)
    assert facts["n_study_runs"] == len(results_runs) - len(appendix)
