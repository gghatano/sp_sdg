"""Run manifest creation and validation (spec section 7)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_KEYS = [
    "run_id",
    "phase",
    "dataset",
    "dataset_checksum",
    "split_checksum",
    "augmentation",
    "augmentation_params",
    "model",
    "model_params",
    "seed",
    "git_commit",
    "git_dirty",
    "python_version",
    "package_lock_checksum",
    "hardware",
    "started_at",
    "ended_at",
    "status",
    "metrics_path",
    "predictions_path",
    "log_path",
]

VALID_STATUSES = {"running", "completed", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Generated / output directories. Changes here are produced by running the
# experiment itself (manifests, metrics, checkpoints under runs/), by data
# download (data/), or by report generation (report/dist/). They must not make
# git_dirty true: the flag exists to record whether the *source* that produced a
# run was committed, so that git_commit uniquely identifies the code + config.
GIT_DIRTY_EXCLUDE = ("runs", "data", "report/dist")


def git_info(repo_root: str | Path = ".") -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root, check=True
        ).stdout.strip()
        # -uno ignores untracked files; the exclude pathspecs drop tracked output
        # dirs so git_dirty reflects only uncommitted *source* (src/config/tests/
        # artifacts/report src). Output churn during the grid does not count.
        excludes = [f":(exclude){p}" for p in GIT_DIRTY_EXCLUDE]
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "-uno", "--", ".", *excludes],
                capture_output=True,
                text=True,
                cwd=repo_root,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", True


def lock_checksum(repo_root: str | Path = ".") -> str:
    lock = Path(repo_root) / "uv.lock"
    if not lock.exists():
        return "missing"
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def hardware_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def new_manifest(
    run_id: str,
    phase: int,
    dataset: str,
    dataset_checksum: str,
    split_checksum: str,
    augmentation: str,
    augmentation_params: dict,
    model: str,
    model_params: dict,
    seed: int,
    repo_root: str | Path = ".",
) -> dict:
    commit, dirty = git_info(repo_root)
    return {
        "run_id": run_id,
        "phase": phase,
        "dataset": dataset,
        "dataset_checksum": dataset_checksum,
        "split_checksum": split_checksum,
        "augmentation": augmentation,
        "augmentation_params": augmentation_params,
        "model": model,
        "model_params": model_params,
        "seed": seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": sys.version.split()[0],
        "package_lock_checksum": lock_checksum(repo_root),
        "hardware": hardware_info(),
        "started_at": utc_now(),
        "ended_at": None,
        "status": "running",
        "metrics_path": None,
        "predictions_path": None,
        "log_path": None,
    }


def validate_manifest(manifest: dict) -> list[str]:
    """Return a list of problems; empty list means valid."""
    problems = [f"missing key: {k}" for k in REQUIRED_KEYS if k not in manifest]
    if manifest.get("status") not in VALID_STATUSES:
        problems.append(f"invalid status: {manifest.get('status')!r}")
    if manifest.get("status") == "completed":
        for k in ("ended_at", "metrics_path", "predictions_path", "log_path"):
            if not manifest.get(k):
                problems.append(f"completed run lacks {k}")
    if not isinstance(manifest.get("seed"), int):
        problems.append("seed must be int")
    return problems


def save_manifest(manifest: dict, manifests_dir: str | Path) -> Path:
    path = Path(manifests_dir) / f"{manifest['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp name per process so two writers of the same run_id cannot
    # clobber each other's tmp mid-write (spec section 7 forbids concurrent
    # writes to one run_id; this hardening keeps a stray double-run from
    # corrupting the store rather than condoning it).
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    os.replace(tmp, path)
    return path


def load_manifest(run_id: str, manifests_dir: str | Path) -> dict | None:
    path = Path(manifests_dir) / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
