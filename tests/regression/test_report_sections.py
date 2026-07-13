"""Regression: the rendered report must contain every required section
(spec 3.6, 9) and never lose them across template edits."""

from signal_aug.reporting.build import REQUIRED_SECTION_IDS, gather_context, render_report


def test_report_contains_required_sections():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    for section_id in REQUIRED_SECTION_IDS:
        assert f'id="{section_id}"' in html, f"missing section: {section_id}"


def test_report_is_japanese_and_self_describing():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    assert 'lang="ja"' in html
    assert "自動生成" in html  # states that results are not hand-typed


def test_references_present():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    assert "UCR Time Series Archive" in html
    assert "MiniRocket" in html


def test_smoke_runs_excluded_from_main_summary():
    """Quality-gate runs on synthetic data must not appear as study results."""
    context = gather_context(".")
    assert all(s["dataset"] != "synthetic" for s in context["summary"])


def test_reference_indices_are_key_based():
    """Inline citations come from the ref map, so inserting a reference cannot
    silently shift citation numbers."""
    context = gather_context(".")
    template = open("report/src/report.template.html").read()
    assert "ref.ucr" in template and "ref.minirocket" in template
    for key in ("ucr", "minirocket", "iwana2021", "aeon"):
        assert key in context["ref"]


def test_findings_rendered_when_present():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    if context["findings"]:
        assert context["findings"][0]["id"] in html
        assert "追試" in html


def test_report_has_paper_and_dashboard_tabs():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    assert 'data-tabbtn="paper"' in html and 'data-tabbtn="dashboard"' in html
    assert 'data-tab="paper"' in html and 'data-tab="dashboard"' in html


def test_paper_tab_has_academic_sections():
    """Paper tab must carry the academic structure; ops ids live in the dashboard."""
    for sid in ("abstract", "introduction", "related-work", "setup", "conclusion"):
        assert sid in REQUIRED_SECTION_IDS
    for sid in ("ops-progress", "ops-audit", "ops-runs", "ops-reproducibility"):
        assert sid in REQUIRED_SECTION_IDS
