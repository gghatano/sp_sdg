"""Subject-count learning-curve runner (Phase 4).

For each (subject_count, repeat, augmentation) it selects whole subjects from
the UCI HAR pool, trains the primary model, and evaluates on the fixed 9-subject
held-out test set. Reuses the manifest/lock machinery of the grid runner.

Leakage rules (spec section 8): the test subjects are disjoint from the pool;
augmentation touches only the selected training subjects; the target metric is
pre-registered (config, artifacts/pre_registration.md).
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from signal_aug.augmentations.methods import apply_augmentation
from signal_aug.data.subject import select_train_subjects, subset_by_subjects, windows_checksum
from signal_aug.data.subject_datasets import load_subject_dataset
from signal_aug.evaluation.metrics import compute_metrics
from signal_aug.experiments import manifest as mf
from signal_aug.experiments.runner import clean_stale_tmp, grid_lock, load_yaml, run_logger
from signal_aug.models import build_model


@dataclass
class SubjectRunSpec:
    run_id: str
    phase: int
    dataset: str
    augmentation: str
    augmentation_params: dict
    model: str
    model_type: str
    model_params: dict
    seed: int
    subject_count: int
    dataset_params: dict = field(default_factory=dict)


def expand_subject_grid(exp_cfg: dict, aug_cfg: dict, model_cfg: dict) -> list[SubjectRunSpec]:
    name = exp_cfg["name"]
    phase = int(exp_cfg["phase"])
    dataset = exp_cfg["dataset"]
    model = exp_cfg["model"]
    model_spec = model_cfg["models"][model]
    dataset_params = dict(exp_cfg.get("dataset_params") or {})
    runs = []
    for n_subj in exp_cfg["subject_counts"]:
        for rep in range(int(exp_cfg["repeats"])):
            for aug in exp_cfg["augmentations"]:
                aug_spec = aug_cfg["augmentations"][aug]
                run_id = f"{name}_{dataset}_n{n_subj}_{aug}_{model}_r{rep}"
                runs.append(
                    SubjectRunSpec(
                        run_id=run_id,
                        phase=phase,
                        dataset=dataset,
                        augmentation=aug_spec["method"],
                        augmentation_params=dict(aug_spec.get("params") or {}),
                        model=model,
                        model_type=model_spec["type"],
                        model_params=dict(model_spec.get("params") or {}),
                        seed=rep,
                        subject_count=int(n_subj),
                        dataset_params=dict(dataset_params),
                    )
                )
    return runs


def execute_subject_run(spec: SubjectRunSpec, pool, test, runs_dir: Path, repo_root: str | Path = ".") -> dict:
    manifests_dir = runs_dir / "manifests"
    log_path = runs_dir / "logs" / f"{spec.run_id}.log"
    metrics_path = runs_dir / "metrics" / f"{spec.run_id}.json"
    predictions_path = runs_dir / "predictions" / f"{spec.run_id}.csv"
    logger = run_logger(spec.run_id, log_path)

    # checksums over the fixed test set and the selected subject subset
    chosen = select_train_subjects(pool.subjects, spec.subject_count, seed=spec.seed)
    pool_dict = {"X_train": pool.X, "y_train": pool.y, "subjects_train": pool.subjects}
    X_train, y_train = subset_by_subjects(pool_dict, chosen)

    # Real content hash of the full pool + test window arrays (not a per-run
    # placeholder), so a silent change in the windowed dataset is visible in the
    # manifest even though the same pool/test is reused across the grid.
    dataset_checksum = f"{spec.dataset.lower()}:{windows_checksum(pool.X, test.X)}"

    manifest = mf.new_manifest(
        run_id=spec.run_id,
        phase=spec.phase,
        dataset=spec.dataset,
        dataset_checksum=dataset_checksum,
        split_checksum="subjects:" + ",".join(map(str, chosen)),
        augmentation=spec.augmentation,
        augmentation_params=spec.augmentation_params,
        model=spec.model,
        model_params=spec.model_params,
        seed=spec.seed,
        repo_root=repo_root,
    )
    manifest["log_path"] = str(log_path)
    manifest["subject_count"] = spec.subject_count
    manifest["subjects_used"] = chosen
    manifest["dataset_params"] = dict(spec.dataset_params)
    mf.save_manifest(manifest, manifests_dir)

    try:
        logger.info("n_subjects=%s subjects=%s train=%s test=%s",
                    spec.subject_count, chosen, X_train.shape, test.X.shape)
        # guard: selected training subjects must be disjoint from test subjects
        if set(chosen) & set(test.subjects.tolist()):
            raise ValueError("subject leak: training subject also in test set")

        X_aug, y_aug = apply_augmentation(
            spec.augmentation, X_train, y_train, seed=spec.seed, params=spec.augmentation_params
        )
        if not np.isfinite(X_aug).all():
            raise ValueError("augmented training data contains NaN/Inf")

        model = build_model(spec.model_type, spec.model_params, seed=spec.seed)
        model.fit(X_aug, y_aug)
        y_pred = model.predict(test.X)
        metrics = compute_metrics(test.y, y_pred)
        metrics["subject_count"] = spec.subject_count
        metrics["n_train_windows"] = int(len(y_train))
        logger.info("metrics=%s", metrics)

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["index,y_true,y_pred"] + [f"{i},{int(t)},{int(p)}" for i, (t, p) in enumerate(zip(test.y, y_pred))]
        predictions_path.write_text("\n".join(lines) + "\n")

        manifest.update(status="completed", ended_at=mf.utc_now(),
                        metrics_path=str(metrics_path), predictions_path=str(predictions_path))
    except Exception:
        logger.error("run failed:\n%s", traceback.format_exc())
        manifest.update(status="failed", ended_at=mf.utc_now())
    finally:
        mf.save_manifest(manifest, manifests_dir)
        for h in logger.handlers:
            h.close()
    return manifest


def run_subject_experiment(
    experiment_config_path: str | Path,
    config_dir: str | Path = "config",
    runs_dir: str | Path = "runs",
    repo_root: str | Path = ".",
    resume: bool = True,
) -> list[dict]:
    config_dir = Path(config_dir)
    exp_cfg = load_yaml(experiment_config_path)
    if not exp_cfg.get("pre_registered"):
        raise ValueError("subject_count config must be pre-registered before running (spec section 5)")
    aug_cfg = load_yaml(config_dir / "augmentations.yaml")
    model_cfg = load_yaml(config_dir / "models.yaml")

    runs = expand_subject_grid(exp_cfg, aug_cfg, model_cfg)
    runs_dir = Path(runs_dir)
    manifests_dir = runs_dir / "manifests"
    results = []
    n_skip = 0

    dataset_params = dict(exp_cfg.get("dataset_params") or {})
    with grid_lock(runs_dir):
        clean_stale_tmp(runs_dir)
        pool, test = load_subject_dataset(exp_cfg["dataset"], **dataset_params)
        for spec in runs:
            existing = mf.load_manifest(spec.run_id, manifests_dir)
            if resume and existing and existing.get("status") == "completed":
                results.append(existing)
                n_skip += 1
                continue
            print(f"[run ] {spec.run_id}")
            results.append(execute_subject_run(spec, pool, test, runs_dir=runs_dir, repo_root=repo_root))

    n_done = sum(1 for r in results if r["status"] == "completed")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    print(f"[done] {n_done} completed ({n_skip} pre-existing), {n_fail} failed, {len(results)} total")
    return results
