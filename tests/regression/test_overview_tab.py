"""Regression: the report opens on an index page that says what each tab holds.

The landing page is the first thing a reader sees on GitHub Pages, so it must
list every tab, link to each one, and carry counts that come from the same
context that built those tabs (an index that drifts from its content is worse
than no index).
"""

import re

from signal_aug.reporting.build import gather_context, render_report


def _html() -> str:
    return render_report(gather_context("."), "report/src", css="")


def test_report_opens_on_the_overview_tab():
    html = _html()
    assert 'data-tabbtn="overview"' in html and 'data-tab="overview"' in html
    assert "showTab(TAB_BY_HASH[location.hash] || 'overview');" in html, "default tab is not the index"
    # the index button comes first in the tab bar
    order = re.findall(r'data-tabbtn="([a-z]+)"', html)
    assert order[0] == "overview", f"tab order starts with {order[0]}"


def test_every_tab_has_a_card_on_the_index():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    tabs = [t for t in re.findall(r'data-tabbtn="([a-z]+)"', html) if t != "overview"]
    carded = [card["tab"] for card in context["overview_cards"]]
    assert carded == tabs, f"index cards {carded} do not match the tabs {tabs}"
    overview = html[html.index('data-tab="overview"'):html.index("PAPER TAB")]
    for tab in tabs:
        assert f'href="#tab-{tab}"' in overview, f"index does not link to {tab}"


def test_cards_describe_and_quantify_their_tab():
    context = gather_context(".")
    for card in context["overview_cards"]:
        assert card["title"] and card["lead"], f"{card['tab']}: card has no description"
        assert card["points"], f"{card['tab']}: card has no contents list"


def test_card_counts_come_from_the_tabs_they_point_at():
    """The numbers on the index must be the ones the tabs actually render."""
    context = gather_context(".")
    cards = {c["tab"]: c for c in context["overview_cards"]}
    text = " ".join(cards["evaluation"]["points"])
    assert str(context["eval_tab"]["n_datasets"]) in text
    assert str(context["eval_tab"]["n_runs"]) in text

    methods_text = " ".join(cards["methods"]["points"])
    n_methods = sum(len(v) for v in context["method_tab"]["entries"].values())
    assert str(n_methods) in methods_text

    appendix_text = " ".join(cards["appendix"]["points"])
    assert str(context["facts"]["n_appendix_runs"]) in appendix_text


def test_index_offers_task_based_entry_points():
    html = _html()
    overview = html[html.index('data-tab="overview"'):html.index("PAPER TAB")]
    assert 'id="overview-start"' in overview
    # the two questions a first-time reader arrives with
    assert "どんなデータを使ったか" in overview
    assert "被験者数を減らせるのか" in overview


def test_landing_page_lists_open_questions_and_queue():
    """A reader (or a collaborator) must be able to see what is still unsettled
    and what is queued without digging through artifacts."""
    context = gather_context(".")
    questions = context["open_questions"].get("questions", [])
    assert questions, "artifacts/open_questions.yaml carries no questions"
    for q in questions:
        for field in ("question", "why", "next", "status", "priority"):
            assert q.get(field), f"{q.get('id')}: missing '{field}'"
        assert q["status"] in {"open", "planned", "blocked"}

    html = render_report(context, "report/src", css="")
    overview = html[html.index('data-tab="overview"'):html.index("PAPER TAB")]
    assert 'id="overview-questions"' in overview
    for q in questions:
        assert q["question"] in overview, f"{q['id']} not rendered"
    # a blocked item states what would unblock it
    blocked = [q for q in questions if q["status"] == "blocked"]
    for q in blocked:
        assert q["next"], f"{q['id']}: blocked without a firing condition"


def test_open_questions_reference_real_findings_and_issues():
    """Cross-references must resolve: a finding id that no longer exists, or an
    issue number typo, makes the list untrustworthy."""
    context = gather_context(".")
    finding_ids = {f["id"] for f in context["findings"]}
    for q in context["open_questions"].get("questions", []):
        for fid in q.get("related") or []:
            assert fid in finding_ids, f"{q['id']}: unknown finding {fid}"
        if q.get("issue"):
            assert isinstance(q["issue"], int), f"{q['id']}: issue must be a number"
