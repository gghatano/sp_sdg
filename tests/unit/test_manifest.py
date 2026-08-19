"""Run manifest schema checks (spec 3.6, 7)."""

import subprocess

from signal_aug.experiments.manifest import (
    REQUIRED_KEYS,
    git_info,
    load_manifest,
    new_manifest,
    save_manifest,
    validate_manifest,
)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_git_dirty_ignores_output_dirs_counts_source(tmp_path):
    """git_dirty must reflect only uncommitted source: churn under runs/ or data/
    (produced by running the grid) is not dirtiness; a source edit is."""
    repo = tmp_path
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "src").mkdir()
    (repo / "src" / "mod.py").write_text("x = 1\n")
    (repo / "runs").mkdir()
    (repo / "runs" / "manifest.json").write_text("{}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")

    # clean tree
    _, dirty = git_info(repo)
    assert dirty is False

    # output-dir churn (as during a grid run) -> still clean
    (repo / "runs" / "manifest.json").write_text('{"run": 1}\n')
    _, dirty = git_info(repo)
    assert dirty is False, "tracked runs/ output changes must not mark source dirty"

    # a real source edit -> dirty
    (repo / "src" / "mod.py").write_text("x = 2\n")
    _, dirty = git_info(repo)
    assert dirty is True


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
