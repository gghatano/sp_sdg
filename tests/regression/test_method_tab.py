"""Regression: the augmentation-method tab (issue #30) must describe every
method the study runs, with the same structure, generated from data rather than
hand-typed HTML (CLAUDE.md: HTML への手入力禁止)."""

import json
from pathlib import Path

import yaml

from signal_aug.augmentations.methods import REGISTRY
from signal_aug.reporting.build import (
    augmentation_example_svg,
    gather_context,
    render_report,
)

REQUIRED_PROSE_FIELDS = ("display", "family", "one_liner", "idea", "procedure",
                         "label", "assumption", "role")


def _profiles() -> dict:
    return yaml.safe_load(Path("references/augmentation_profiles.yaml").read_text(encoding="utf-8"))


def _config_methods() -> set:
    cfg = yaml.safe_load(Path("config/augmentations.yaml").read_text(encoding="utf-8"))
    return set(cfg["augmentations"])


def test_every_implemented_method_is_described():
    """A method in the registry but missing from the tab would be invisible to
    the reader; a described method that no longer exists would be a stale doc."""
    described = set(_profiles()["methods"])
    assert described == set(REGISTRY), (
        f"described {sorted(described)} != implemented {sorted(REGISTRY)}"
    )
    assert described == _config_methods()


def test_all_entries_share_the_same_structure():
    profiles = _profiles()
    for key, entry in profiles["methods"].items():
        for field in REQUIRED_PROSE_FIELDS:
            assert entry.get(field), f"{key}: missing '{field}'"
        assert entry["family"] in profiles["families"], f"{key}: unknown family {entry['family']}"
        assert "params" in entry, f"{key}: missing 'params' (may be empty for none)"


def test_parameter_values_come_from_config_not_prose():
    """The prose names each knob; the VALUE must come from config/augmentations.yaml
    so the tab cannot drift from what the experiments actually used."""
    cfg = yaml.safe_load(Path("config/augmentations.yaml").read_text(encoding="utf-8"))["augmentations"]
    tab = gather_context(".")["method_tab"]
    entries = {e["key"]: e for family in tab["entries"].values() for e in family}
    for key, entry in entries.items():
        configured = cfg[key].get("params", {})
        for param in entry["params"]:
            assert param["name"] in configured, f"{key}: '{param['name']}' not in config"
            assert param["value"] == configured[param["name"]]
        # every configured knob must be explained
        described = {p["name"] for p in entry["params"]}
        assert described == set(configured), f"{key}: params {described} != config {set(configured)}"


def test_method_tab_rendered_with_sections():
    context = gather_context(".")
    html = render_report(context, "report/src", css="")
    assert 'data-tabbtn="methods"' in html and 'data-tab="methods"' in html
    for family in ("magnitude", "pattern", "trivial", "control"):
        assert f'id="methods-{family}"' in html
    for key in REGISTRY:
        assert f'id="aug-{key.replace("_", "-")}"' in html, f"missing method article: {key}"


def test_measured_effects_are_wired_from_results():
    """Each method's effect table must come from results.json stats, not prose."""
    context = gather_context(".")
    tab = context["method_tab"]
    entries = {e["key"]: e for family in tab["entries"].values() for e in family}
    stats_augs = {s["augmentation"] for s in context["stats"] if s.get("metric") == "accuracy"}
    for key in stats_augs:
        assert entries[key]["effects"], f"{key}: no measured effect wired"
        for effect in entries[key]["effects"]:
            assert effect["model"] and effect["p_value"] is not None


def test_examples_generated_by_the_real_augmenters():
    """The before/after traces must be produced by running the production
    augmenters with the configured parameters (no illustrative hand-drawing)."""
    payload = json.loads(
        Path("report/assets/data/augmentation_examples.json").read_text(encoding="utf-8")
    )
    assert payload["methods"], "augmentation_examples.json is empty"
    cfg = yaml.safe_load(Path("config/augmentations.yaml").read_text(encoding="utf-8"))["augmentations"]
    for key, entry in payload["methods"].items():
        assert entry["params"] == cfg[key].get("params", {}), f"{key}: params drifted from config"
        assert entry["n_after"] >= entry["n_original"]
        if key == "none":
            assert entry["n_synthetic"] == 0
        else:
            assert entry["n_synthetic"] > 0, f"{key}: produced no synthetic samples"


def test_label_shuffle_is_marked_as_a_control():
    profiles = _profiles()
    assert profiles["methods"]["label_shuffle"]["family"] == "control"
    # the corrected interpretation (issue #23) must be stated, not the old
    # "zero class signal floor" reading
    deviation = profiles["methods"]["label_shuffle"]["deviation"] or ""
    assert "床" in deviation or "悲観的対照" in deviation


def test_example_svg_handles_empty_input():
    assert augmentation_example_svg([], None, "#000") == ""
