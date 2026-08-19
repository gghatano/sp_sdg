"""Regression: the paper tab is scoped to "survey explanation + partial
reproduction", and the two things issue #31 asked to split out — per-dataset
learning/evaluation, and the subject-count reduction study — live in their own
tabs. Guards against the sections silently drifting back into the paper."""

from pathlib import Path

import yaml

from signal_aug.augmentations.methods import REGISTRY
from signal_aug.reporting.build import REQUIRED_SECTION_IDS, gather_context, render_report

REDUCTION_SECTIONS = (
    "subject-reduction",
    "subject-reduction-wisdm",
    "subject-reduction-cross",
    "subject-reduction-wesad",
    "subject-reduction-synthesis",
)


def _html() -> str:
    return render_report(gather_context("."), "report/src", css="")


def _tab_body(html: str, tab: str) -> str:
    """The markup of one tab (tabs are sibling blocks keyed by data-tab)."""
    start = html.index(f'data-tab="{tab}"')
    rest = html[start + 1:]
    nxt = rest.find('data-tab="')
    return rest[:nxt] if nxt != -1 else rest


def test_new_tabs_exist():
    html = _html()
    for tab in ("evaluation", "reduction"):
        assert f'data-tabbtn="{tab}"' in html and f'data-tab="{tab}"' in html


def test_reduction_sections_moved_out_of_the_paper_tab():
    """RQ2 content must be reachable, but not inside the paper tab any more."""
    html = _html()
    paper = _tab_body(html, "paper")
    reduction = _tab_body(html, "reduction")
    for sid in REDUCTION_SECTIONS:
        assert f'id="{sid}"' in reduction, f"{sid} missing from the reduction tab"
        assert f'id="{sid}"' not in paper, f"{sid} still in the paper tab"


def test_per_dataset_evaluation_is_its_own_tab():
    """issue #31: 何を学習して何を評価しているのかがデータセットごとに分かること。"""
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    evaluation = _tab_body(html, "evaluation")
    assert 'id="eval-protocol"' in evaluation
    for row in context["eval_tab"]["rows"]:
        assert f'id="{row["anchor"]}"' in evaluation, f"missing evaluation article: {row['key']}"
        # each dataset states its input, its split and its run count
        assert row["input_shape"]["channels"] and row["input_shape"]["length"]
        assert row["input_shape"]["n_classes"]
        assert row["split"] and row["models"] and row["n_runs"] > 0


def test_evaluation_rows_cover_every_dataset_with_runs():
    context = gather_context(".")
    with_runs = {r["dataset"] for r in context["results"]["runs"]
                 if r["status"] == "completed" and r["dataset"] != "synthetic"}
    covered = {r["key"] for r in context["eval_tab"]["rows"]}
    assert covered == with_runs, f"evaluation tab covers {sorted(covered)} != runs {sorted(with_runs)}"


def test_evaluation_numbers_come_from_results():
    """Per-dataset tables must be aggregated from results.json, not hand-typed."""
    context = gather_context(".")
    rows = {r["key"]: r for r in context["eval_tab"]["rows"]}
    summary = context["results"]["summary"]
    for key, row in rows.items():
        expected = [s for s in summary
                    if s["dataset"] == key and s.get("train_fraction", 1.0) == 1.0]
        assert len(row["summary"]) == len(expected), f"{key}: summary rows drifted"
    assert context["eval_tab"]["n_runs"] == sum(r["n_runs"] for r in rows.values())


def test_survey_section_states_coverage_and_gaps():
    """The paper is framed as a partial reproduction: the survey's taxonomy must
    be shown with what was NOT evaluated, not only what was."""
    context = gather_context(".")
    survey = context["survey_tab"]
    assert "survey" in REQUIRED_SECTION_IDS
    assert survey["taxonomy"] and survey["claims"]
    covered = [g for g in survey["taxonomy"] if g["our_coverage"] == "partial"]
    uncovered = [g for g in survey["taxonomy"] if g["our_coverage"] == "none"]
    assert covered and uncovered, "coverage table must show both sides"
    for group in survey["taxonomy"]:
        assert group["not_covered"], f"{group['key']}: untested scope must be stated"
    # claims must be honest about what was not tested
    assert any(c["our_status"] == "not_tested" for c in survey["claims"])
    html = render_report(context, "report/src", css="")
    assert 'id="survey"' in _tab_body(html, "paper")


def test_survey_methods_match_the_implemented_registry():
    """The coverage table names real methods, so it cannot claim a method the
    study never ran."""
    data = yaml.safe_load(Path("references/survey_overview.yaml").read_text(encoding="utf-8"))
    named = {m for g in data["taxonomy"] for m in g.get("our_methods", [])}
    assert named <= set(REGISTRY), f"unknown methods in survey coverage: {named - set(REGISTRY)}"


def test_survey_claim_evidence_points_at_real_findings():
    context = gather_context(".")
    finding_ids = {f["id"] for f in context["findings"]}
    for claim in context["survey_tab"]["claims"]:
        for fid in claim.get("our_evidence", []):
            assert fid in finding_ids, f"{claim['id']}: unknown finding {fid}"


def test_reading_guide_links_every_tab():
    """A reader landing on the paper tab must be told where the rest lives."""
    paper = _tab_body(_html(), "paper")
    for tab in ("datasets", "methods", "evaluation", "reduction", "dashboard", "repro"):
        assert f"showTab('{tab}')" in paper, f"reading guide does not link {tab}"
