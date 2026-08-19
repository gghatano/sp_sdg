"""Experiment runner: expands a grid config into runs, executes them with
manifests, and supports resume (completed runs are skipped).

Leakage rules enforced here (spec section 8):
- augmentation is applied to the training split only
- the model never sees test data before predict()
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from signal_aug.augmentations.methods import apply_augmentation
from signal_aug.data.loader import DatasetSplits, load_dataset, stratified_subsample
from signal_aug.evaluation.metrics import compute_metrics
from signal_aug.experiments import manifest as mf
from signal_aug.models import build_model


@dataclass
class RunSpec:
    run_id: str
    phase: int
    dataset: str
    augmentation: str
    augmentation_params: dict
    model: str
    model_type: str
    model_params: dict
    seed: int
    train_fraction: float = 1.0


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid exists and is signalable. A live pid we
    do not own raises PermissionError — still alive, so return True."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # Windows: os.kill(pid, 0) on a non-existent pid raises OSError with
        # winerror 87 (ERROR_INVALID_PARAMETER) instead of ProcessLookupError.
        if getattr(exc, "winerror", None) == 87:
            return False
        raise
    return True


@contextmanager
def grid_lock(runs_dir: Path):
    """Prevent two runner processes from executing grids concurrently — the
    root cause of run_id write races (spec section 7). Acquisition is atomic
    via O_CREAT|O_EXCL so two processes cannot both believe they hold the lock
    (the previous exists()-then-write left a TOCTOU window). Stale locks whose
    pid is dead are reclaimed."""
    lock = Path(runs_dir) / ".runner.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload = str(os.getpid())

    def _try_create() -> bool:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        return True

    if not _try_create():
        # lock exists: reclaim only if the holder pid is dead
        try:
            old_pid = int(lock.read_text(encoding="utf-8").strip())
            holder_alive = _pid_alive(old_pid)
        except ValueError:
            old_pid, holder_alive = None, False  # corrupt lock, treat as stale
        if holder_alive:
            raise RuntimeError(
                f"another runner (pid {old_pid}) holds {lock}. "
                f"Wait for it, or remove the lock if that process is dead."
            )
        lock.unlink(missing_ok=True)
        if not _try_create():
            raise RuntimeError(f"could not acquire {lock} after reclaiming a stale lock")

    try:
        yield
    finally:
        # only the owner removes the lock
        if lock.exists() and lock.read_text(encoding="utf-8").strip() == payload:
            lock.unlink(missing_ok=True)


def clean_stale_tmp(runs_dir: Path) -> int:
    """Remove orphaned manifest tmp files left by a process that died between
    write and atomic replace. Safe to call while holding the grid lock (no other
    runner is writing). Returns the number removed."""
    n = 0
    for tmp in Path(runs_dir).glob("manifests/*.json.tmp.*"):
        tmp.unlink(missing_ok=True)
        n += 1
    return n


def expand_grid(experiment_cfg: dict, aug_cfg: dict, model_cfg: dict) -> list[RunSpec]:
    runs = []
    phase = int(experiment_cfg["phase"])
    name = experiment_cfg["name"]
    fractions = [float(f) for f in experiment_cfg.get("train_fractions", [1.0])]
    for dataset in experiment_cfg["datasets"]:
        for fraction in fractions:
            # fraction omitted from run_id at 1.0 so Phase 1 run_ids stay stable
            frac_tag = "" if fraction == 1.0 else f"_f{int(round(fraction * 100))}"
            for aug in experiment_cfg["augmentations"]:
                aug_spec = aug_cfg["augmentations"][aug]
                for model in experiment_cfg["models"]:
                    model_spec = model_cfg["models"][model]
                    for seed in experiment_cfg["seeds"]:
                        run_id = f"{name}_{dataset}{frac_tag}_{aug}_{model}_s{seed}"
                        runs.append(
                            RunSpec(
                                run_id=run_id,
                                phase=phase,
                                dataset=dataset,
                                augmentation=aug_spec["method"],
                                augmentation_params=dict(aug_spec.get("params") or {}),
                                model=model,
                                model_type=model_spec["type"],
                                model_params=dict(model_spec.get("params") or {}),
                                seed=int(seed),
                                train_fraction=fraction,
                            )
                        )
    return runs


def run_logger(run_id: str, log_path: Path) -> logging.Logger:
    """Per-run file logger. Closes any pre-existing handlers before replacing
    them so re-running a run_id in-process does not leak file descriptors."""
    logger = logging.getLogger(f"run.{run_id}")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


def execute_run(spec: RunSpec, data: DatasetSplits, runs_dir: str | Path = "runs", repo_root: str | Path = ".") -> dict:
    runs_dir = Path(runs_dir)
    manifests_dir = runs_dir / "manifests"
    log_path = runs_dir / "logs" / f"{spec.run_id}.log"
    metrics_path = runs_dir / "metrics" / f"{spec.run_id}.json"
    predictions_path = runs_dir / "predictions" / f"{spec.run_id}.csv"

    logger = run_logger(spec.run_id, log_path)
    manifest = mf.new_manifest(
        run_id=spec.run_id,
        phase=spec.phase,
        dataset=spec.dataset,
        dataset_checksum=data.dataset_checksum,
        split_checksum=data.split_checksum,
        augmentation=spec.augmentation,
        augmentation_params=spec.augmentation_params,
        model=spec.model,
        model_params=spec.model_params,
        seed=spec.seed,
        repo_root=repo_root,
    )
    manifest["log_path"] = str(log_path)
    manifest["train_fraction"] = spec.train_fraction
    mf.save_manifest(manifest, manifests_dir)

    try:
        logger.info("dataset=%s train=%s test=%s", spec.dataset, data.X_train.shape, data.X_test.shape)
        X_train, y_train = stratified_subsample(
            data.X_train, data.y_train, spec.train_fraction, seed=spec.seed
        )
        manifest["n_train_used"] = int(len(y_train))
        if spec.train_fraction != 1.0:
            logger.info("train_fraction=%s subsampled=%s", spec.train_fraction, X_train.shape)
        X_aug, y_aug = apply_augmentation(
            spec.augmentation, X_train, y_train, seed=spec.seed, params=spec.augmentation_params
        )
        if not np.isfinite(X_aug).all():
            raise ValueError("augmented training data contains NaN/Inf")
        logger.info("augmentation=%s train_after=%s", spec.augmentation, X_aug.shape)

        model = build_model(spec.model_type, spec.model_params, seed=spec.seed)
        model.fit(X_aug, y_aug)
        y_pred = model.predict(data.X_test)
        metrics = compute_metrics(data.y_test, y_pred)
        # train accuracy on the original (pre-augmentation) training subsample, for
        # the generalization-gap analysis (train_accuracy - accuracy); see H2 (issue #12)
        from sklearn.metrics import accuracy_score

        metrics["train_accuracy"] = float(accuracy_score(y_train, model.predict(X_train)))
        logger.info("metrics=%s", metrics)

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pred_lines = ["index,y_true,y_pred"] + [
            f"{i},{int(t)},{int(p)}" for i, (t, p) in enumerate(zip(data.y_test, y_pred))
        ]
        predictions_path.write_text("\n".join(pred_lines) + "\n")

        manifest.update(
            status="completed",
            ended_at=mf.utc_now(),
            metrics_path=str(metrics_path),
            predictions_path=str(predictions_path),
        )
    except Exception:
        logger.error("run failed:\n%s", traceback.format_exc())
        manifest.update(status="failed", ended_at=mf.utc_now())
    finally:
        mf.save_manifest(manifest, manifests_dir)
        for h in logger.handlers:
            h.close()
    return manifest


def run_experiment(
    experiment_config_path: str | Path,
    config_dir: str | Path = "config",
    runs_dir: str | Path = "runs",
    data_dir: str | Path = "data/raw",
    repo_root: str | Path = ".",
    resume: bool = True,
) -> list[dict]:
    config_dir = Path(config_dir)
    experiment_cfg = load_yaml(experiment_config_path)
    if experiment_cfg.get("status") == "draft":
        raise ValueError(f"experiment config {experiment_config_path} is a draft; finalize it first")
    aug_cfg = load_yaml(config_dir / "augmentations.yaml")
    model_cfg = load_yaml(config_dir / "models.yaml")
    datasets_cfg = load_yaml(config_dir / "datasets.yaml")

    runs = expand_grid(experiment_cfg, aug_cfg, model_cfg)
    manifests_dir = Path(runs_dir) / "manifests"
    results = []
    data_cache: dict[str, DatasetSplits] = {}

    n_skip = 0
    with grid_lock(Path(runs_dir)):
        clean_stale_tmp(Path(runs_dir))
        for spec in runs:
            existing = mf.load_manifest(spec.run_id, manifests_dir)
            if resume and existing and existing.get("status") == "completed":
                results.append(existing)
                n_skip += 1
                continue
            if spec.dataset not in data_cache:
                data_cache[spec.dataset] = load_dataset(spec.dataset, datasets_cfg, data_dir=data_dir)
            print(f"[run ] {spec.run_id}")
            results.append(execute_run(spec, data_cache[spec.dataset], runs_dir=runs_dir, repo_root=repo_root))

    n_done = sum(1 for r in results if r["status"] == "completed")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    print(f"[done] {n_done} completed ({n_skip} pre-existing), {n_fail} failed, {len(results)} total")
    return results
