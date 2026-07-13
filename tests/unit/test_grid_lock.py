"""The grid lock prevents concurrent runners writing the same run_id."""

import os

import pytest

from signal_aug.experiments.runner import grid_lock


def test_lock_acquired_and_released(tmp_path):
    with grid_lock(tmp_path):
        assert (tmp_path / ".runner.lock").exists()
    assert not (tmp_path / ".runner.lock").exists()


def test_second_live_lock_is_refused(tmp_path):
    with grid_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another runner"):
            with grid_lock(tmp_path):
                pass


def test_stale_lock_is_reclaimed(tmp_path):
    # a pid that cannot be alive
    (tmp_path / ".runner.lock").write_text("999999999")
    with grid_lock(tmp_path):
        assert (tmp_path / ".runner.lock").read_text().strip() == str(os.getpid())


def test_corrupt_lock_is_reclaimed(tmp_path):
    # non-numeric lock payload must be treated as stale, not crash
    (tmp_path / ".runner.lock").write_text("not-a-pid")
    with grid_lock(tmp_path):
        assert (tmp_path / ".runner.lock").read_text().strip() == str(os.getpid())


def test_clean_stale_tmp_removes_orphaned_tmp_files(tmp_path):
    from signal_aug.experiments.runner import clean_stale_tmp

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "run_a.json").write_text("{}")             # real manifest, keep
    (manifests / "run_b.json.tmp.123").write_text("{}")     # orphan, remove
    (manifests / "run_c.json.tmp.456").write_text("{}")     # orphan, remove
    removed = clean_stale_tmp(tmp_path)
    assert removed == 2
    assert (manifests / "run_a.json").exists()
    assert not list(manifests.glob("*.tmp.*"))
