"""Regression: the dataset tab (issue #29) must exist, describe every dataset
the study uses with the same structure, and be generated from data rather than
hand-typed into the HTML (CLAUDE.md: HTML への手入力禁止)."""

import json
from pathlib import Path

import yaml

from signal_aug.reporting.build import (
    class_balance_svg,
    channel_stack_svg,
    gather_context,
    render_report,
    waveform_svg,
)

# every dataset the study actually ran on must be described in the tab
EXPECTED_DATASETS = {
    "ECG5000", "FordA", "GunPoint", "ECG200", "TwoLeadECG", "ItalyPowerDemand",
    "MoteStrain", "SonyAIBORobotSurface1", "CBF", "Plane", "ArrowHead", "Coffee",
    "Wafer", "UCI_HAR", "WISDM", "PAMAP2", "WESAD",
}
# same-structure requirement: these prose fields are mandatory for every entry
REQUIRED_PROSE_FIELDS = ("display", "group", "role", "what", "task", "schema", "classes", "caveats")


def _profiles() -> dict:
    return yaml.safe_load(Path("references/dataset_profiles.yaml").read_text(encoding="utf-8"))


def test_every_study_dataset_is_described():
    described = set(_profiles()["datasets"])
    missing = EXPECTED_DATASETS - described
    assert not missing, f"datasets missing from the dataset tab: {sorted(missing)}"


def test_all_entries_share_the_same_structure():
    for key, entry in _profiles()["datasets"].items():
        for field in REQUIRED_PROSE_FIELDS:
            assert entry.get(field), f"{key}: missing '{field}'"
        assert entry["group"] in _profiles()["groups"], f"{key}: unknown group {entry['group']}"


def test_prose_carries_no_counts():
    """Counts must come from data/metadata + generated samples, so the prose file
    is not a second, drifting source of numbers."""
    profiles = _profiles()["datasets"]
    for key, entry in profiles.items():
        for field in ("what", "task", "schema"):
            text = entry.get(field) or ""
            assert "件数=" not in text, f"{key}.{field} appears to hard-code a count"


def test_dataset_tab_rendered_with_sections():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    assert 'data-tabbtn="datasets"' in html and 'data-tab="datasets"' in html
    assert 'id="datasets-ucr"' in html and 'id="datasets-subject"' in html
    for key in EXPECTED_DATASETS:
        anchor = f'id="dataset-{key.lower().replace("_", "-")}"'
        assert anchor in html, f"missing dataset article: {key}"


def test_dataset_tab_context_merges_three_sources():
    tab = gather_context(".")["dataset_tab"]
    entries = {e["key"]: e for group in tab["entries"].values() for e in group}
    har = entries["UCI_HAR"]
    assert har["meta"].get("channels"), "loader metadata (data/metadata/*.json) not merged"
    assert har["what"], "prose (references/dataset_profiles.yaml) not merged"
    assert har["license"]
    # subject datasets whose samples are committed must carry real traces
    assert har["sample"].get("traces"), "generated samples (dataset_samples.json) not merged"
    assert har["waveform_svg"].startswith("<svg")


def test_class_descriptions_match_the_real_class_names():
    """A class described in prose but absent from the data (or renamed) shows up
    as a missing count; catch that here rather than in a reader's eyes."""
    tab = gather_context(".")["dataset_tab"]
    for group in tab["entries"].values():
        for entry in group:
            if not entry["sample"].get("class_distribution"):
                continue  # no committed samples (PAMAP2/WESAD)
            counts = {d["class_name"] for d in entry["sample"]["class_distribution"]}
            described = {str(c["name"]) for c in entry["classes"]}
            assert described == counts, (
                f"{entry['key']}: class names in prose {sorted(described)} != data {sorted(counts)}"
            )


def test_samples_are_committed_and_train_only():
    """The generated sample file must be present (so the report builds without
    raw data) and must not carry test-split values (spec section 8)."""
    samples = json.loads(Path("report/assets/data/dataset_samples.json").read_text(encoding="utf-8"))
    assert samples["datasets"], "dataset_samples.json is empty"
    for key, entry in samples["datasets"].items():
        assert "n_test_windows" not in entry, f"{key}: test-split values must not be sampled"
        for trace in entry.get("traces", []):
            assert len(trace["values"]) <= samples["max_points"]


def test_restricted_datasets_have_no_committed_signal_values():
    """WESAD grants no redistribution and PAMAP2 is skipped by cost: neither may
    appear in the committed sample file."""
    samples = json.loads(Path("report/assets/data/dataset_samples.json").read_text(encoding="utf-8"))
    assert "WESAD" not in samples["datasets"]
    assert "PAMAP2" not in samples["datasets"]


def test_svg_helpers_handle_empty_input():
    assert waveform_svg([]) == ""
    assert channel_stack_svg([]) == ""
    assert class_balance_svg([]) == ""
