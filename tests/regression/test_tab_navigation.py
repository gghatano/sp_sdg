"""Regression: links between tabs must actually navigate.

A plain "#id" anchor cannot scroll to a target inside a hidden tab, so the
report resolves the owning tab first (gotoSection). These tests guard the two
failure modes: a link that points at nothing, and the loss of the handler that
makes cross-tab links work at all.
"""

import re

from signal_aug.reporting.build import gather_context, render_report


def _html() -> str:
    return render_report(gather_context("."), "report/src", css="")


def test_every_internal_anchor_has_a_target():
    """No dead "#..." link anywhere in the report."""
    html = _html()
    ids = set(re.findall(r'id="([^"]+)"', html))
    targets = {h for h in re.findall(r'href="#([^"]+)"', html) if h}
    tab_links = {t for t in targets if t.startswith("tab-")}
    dead = {t for t in targets - tab_links if t not in ids}
    assert not dead, f"links with no target section: {sorted(dead)}"
    tabs = set(re.findall(r'data-tabbtn="([^"]+)"', html))
    dead_tabs = {t for t in tab_links if t[len("tab-"):] not in tabs}
    assert not dead_tabs, f"links to unknown tabs: {sorted(dead_tabs)}"


def test_cross_tab_navigation_handler_is_present():
    html = _html()
    assert "function gotoSection(" in html
    # a delegated click handler resolves the owning tab before scrolling
    assert 'a[href^="#"]' in html
    assert "closest('[data-tab]')" in html


def test_dataset_and_evaluation_tabs_link_to_each_other():
    """The link the reader clicks from 「学習と評価」 must reach the dataset
    description, and back."""
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    ids = set(re.findall(r'id="([^"]+)"', html))
    for row in context["eval_tab"]["rows"]:
        dataset_anchor = f"dataset-{row['key'].lower().replace('_', '-')}"
        assert f'href="#{dataset_anchor}"' in html, f"{row['key']}: no link to its data description"
        assert dataset_anchor in ids, f"{row['key']}: dataset anchor missing"
        assert f'href="#{row["anchor"]}"' in html, f"{row['key']}: no link back from the dataset tab"


def test_deep_link_to_a_section_opens_its_tab():
    html = _html()
    # on load, a "#section" hash resolves through gotoSection (not only #tab-*)
    assert "if (h && !TAB_BY_HASH[location.hash] && document.getElementById(h)) gotoSection(h);" in html
