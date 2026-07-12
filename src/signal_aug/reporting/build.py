"""Build the static HTML report (report/dist/index.html).

Pipeline: results.json + config + artifacts -> Jinja2 template -> Tailwind CSS
build -> self-contained offline HTML. Report content is fully data-driven;
nothing is hand-typed into the HTML (spec sections 3.10, 9).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

PHASE_NAMES = {
    0: "Phase 0: 基盤構築",
    1: "Phase 1: UCR最小追試",
    2: "Phase 2: UCR横断比較",
    3: "Phase 3: 被験者ID付きデータ選定",
    4: "Phase 4: 被験者数学習曲線",
    5: "Phase 5: 被験者数削減評価",
    6: "Phase 6: 手法改善・別データ検証",
    7: "Phase 7: 統合レポート・研究成果化",
}

REQUIRED_SECTION_IDS = [
    "exec-summary",
    "purpose",
    "current-phase",
    "progress",
    "reproduction-conditions",
    "datasets",
    "augmentations",
    "models",
    "reproducibility",
    "results",
    "paper-comparison",
    "failed-runs",
    "audit",
    "limitations",
    "next-tasks",
    "references",
]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _markdown_bullets(path: Path) -> list[str]:
    """Extract top-level bullet items from a markdown file."""
    if not path.exists():
        return []
    return [
        line.strip()[2:].strip()
        for line in path.read_text().splitlines()
        if line.strip().startswith("- ")
    ]


def gather_context(repo_root: str | Path = ".") -> dict:
    root = Path(repo_root)
    results = _load_json(root / "report/assets/data/results.json") or {
        "runs": [],
        "summary": [],
        "failed_runs": [],
        "audit": None,
    }
    task_queue = _load_yaml(root / "artifacts/task_queue.yaml")
    tasks = task_queue.get("tasks", [])
    current_phase = task_queue.get("current_phase", 0)
    references = _load_json(root / "report/assets/data/references.json") or []

    completed_runs = [r for r in results["runs"] if r["status"] == "completed"]
    reproducibility = {}
    if completed_runs:
        latest = max(completed_runs, key=lambda r: r.get("ended_at") or "")
        reproducibility = {
            "git_commit": latest.get("git_commit", ""),
            "python_version": latest.get("python_version", ""),
            "n_dirty_runs": sum(1 for r in completed_runs if r.get("git_dirty")),
            "n_completed": len(completed_runs),
        }

    # smoke runs on synthetic data are quality-gate checks, not study results
    summary_main = [s for s in results["summary"] if s["dataset"] != "synthetic"]

    baseline = {
        (s["dataset"], s.get("train_fraction", 1.0), s["model"]): s
        for s in summary_main
        if s["augmentation"] == "none"
    }
    for s in summary_main:
        base = baseline.get((s["dataset"], s.get("train_fraction", 1.0), s["model"]))
        s["delta_vs_none"] = (
            round(s["accuracy_mean"] - base["accuracy_mean"], 4) if base and s["augmentation"] != "none" else None
        )
        s["baseline_std"] = base["accuracy_std"] if base else None

    deltas = sorted(
        [s for s in summary_main if s.get("delta_vs_none") is not None],
        key=lambda s: s["delta_vs_none"],
        reverse=True,
    )
    best_improvements = deltas[:3]
    worst_degradations = list(reversed(deltas[-3:])) if deltas else []

    findings_data = _load_json(root / "artifacts/findings.json") or {}
    references_index = {r["key"]: i + 1 for i, r in enumerate(references)}

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "results": results,
        "summary": summary_main,
        "best_improvements": best_improvements,
        "worst_degradations": worst_degradations,
        "findings": findings_data.get("findings", []),
        "ref": references_index,
        "n_runs": len(results["runs"]),
        "n_completed": len(completed_runs),
        "n_failed": len(results["failed_runs"]),
        "tasks": tasks,
        "tasks_done": [t for t in tasks if t.get("status") == "done"],
        "tasks_doing": [t for t in tasks if t.get("status") == "in_progress"],
        "tasks_todo": [t for t in tasks if t.get("status") == "todo"],
        "current_phase": current_phase,
        "current_phase_name": PHASE_NAMES.get(current_phase, f"Phase {current_phase}"),
        "datasets_cfg": _load_yaml(root / "config/datasets.yaml").get("datasets", {}),
        "augmentations_cfg": _load_yaml(root / "config/augmentations.yaml").get("augmentations", {}),
        "models_cfg": _load_yaml(root / "config/models.yaml").get("models", {}),
        "reproduction_targets": _load_yaml(root / "references/reproduction_targets.yaml"),
        "limitations": _markdown_bullets(root / "artifacts/limitations.md"),
        "audit": results.get("audit"),
        "reproducibility": reproducibility,
        "references": references,
    }


def render_report(context: dict, template_dir: str | Path = "report/src", css: str = "") -> str:
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("report.template.html")
    return template.render(css=css, **context)


def build_css(repo_root: str | Path = ".", rendered_html_path: Path | None = None) -> str:
    """Run the Tailwind CLI over the rendered HTML. Falls back to the last
    built CSS if node_modules is unavailable (keeps CI/network-free builds working)."""
    root = Path(repo_root)
    report_dir = root / "report"
    css_cache = report_dir / "dist/assets/report.css"
    tailwind_bin = report_dir / "node_modules/.bin/tailwindcss"
    if tailwind_bin.exists() and shutil.which("node"):
        out = css_cache
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(tailwind_bin.resolve()),
                "-i", "src/input.css",
                "-o", str(out.relative_to(report_dir)),
                "--content", str(rendered_html_path.relative_to(report_dir)) if rendered_html_path else "dist/index.html",
                "--minify",
            ],
            cwd=report_dir,
            check=True,
            capture_output=True,
        )
    if css_cache.exists():
        return css_cache.read_text()
    return ""  # unstyled but valid HTML


def build_report(repo_root: str | Path = ".") -> Path:
    root = Path(repo_root)
    context = gather_context(root)
    dist = root / "report/dist"
    dist.mkdir(parents=True, exist_ok=True)

    # two-pass: render for Tailwind content scan, then inline the built CSS
    tmp = root / "report/dist/assets/index.tmp.html"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(render_report(context, root / "report/src", css=""))
    css = build_css(root, rendered_html_path=tmp)
    tmp.unlink(missing_ok=True)

    out = dist / "index.html"
    out.write_text(render_report(context, root / "report/src", css=css))
    return out
