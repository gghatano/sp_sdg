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
