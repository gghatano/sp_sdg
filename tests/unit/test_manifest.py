"""Run manifest schema checks (spec 3.6, 7)."""

from signal_aug.experiments.manifest import (
    REQUIRED_KEYS,
    load_manifest,
    new_manifest,
    save_manifest,
    validate_manifest,
)


def make_manifest():
    return new_manifest(
        run_id="test_run",
        phase=0,
        dataset="synthetic",
        dataset_checksum="abc",
        split_checksum="def",
        augmentation="none",
        augmentation_params={},
        model="cnn1d_smoke",
        model_params={},
        seed=0,
    )


def test_new_manifest_has_all_required_keys():
    manifest = make_manifest()
    assert all(k in manifest for k in REQUIRED_KEYS)
    assert validate_manifest(manifest) == []


def test_completed_requires_paths():
    manifest = make_manifest()
    manifest["status"] = "completed"
    problems = validate_manifest(manifest)
    assert any("ended_at" in p for p in problems)
    assert any("metrics_path" in p for p in problems)


def test_invalid_status_detected():
    manifest = make_manifest()
    manifest["status"] = "banana"
    assert any("invalid status" in p for p in validate_manifest(manifest))


def test_save_and_load_roundtrip(tmp_path):
    manifest = make_manifest()
    save_manifest(manifest, tmp_path)
    loaded = load_manifest("test_run", tmp_path)
    assert loaded == manifest
