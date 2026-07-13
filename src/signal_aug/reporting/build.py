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

# Single source of augmentation colors. Any augmentation not listed falls back
# to _FALLBACK_COLOR, and charts enumerate augmentations present in the data
# (not this dict), so a new method never silently vanishes from a figure.
AUG_COLORS = {
    "none": "#64748b",
    "oversample": "#0ea5e9",
    "jitter": "#22c55e",
    "scaling": "#eab308",
    "mixup": "#ec4899",
    "dtw": "#8b5cf6",
    "smote": "#f97316",
    "label_shuffle": "#94a3b8",
}
_FALLBACK_COLOR = "#334155"


def aug_color(aug: str) -> str:
    return AUG_COLORS.get(aug, _FALLBACK_COLOR)


def learning_curve_svg(dataset: str, model: str, curves: dict, width: int = 320, height: int = 200) -> str:
    """Inline SVG line chart of accuracy vs train_fraction, one line per
    augmentation. Self-contained (no JS/external refs) for offline viewing."""
    pad_l, pad_r, pad_t, pad_b = 40, 8, 12, 28
    prefix = f"{dataset}|{model}|"
    # enumerate augmentations present in the data (not AUG_COLORS) so a method
    # without a preassigned color still appears
    series = {
        key[len(prefix):]: pts
        for key, pts in sorted(curves.items())
        if key.startswith(prefix)
    }
    if not series:
        return ""
    all_pts = [p for pts in series.values() for p in pts]
    ys = [p["accuracy_mean"] for p in all_pts]
    y_min, y_max = min(ys), max(ys)
    if y_max - y_min < 0.02:
        y_min, y_max = y_min - 0.01, y_max + 0.01
    fracs = sorted({p["train_fraction"] for p in all_pts})
    x_min, x_max = min(fracs), max(fracs)

    def sx(f: float) -> float:
        return pad_l + (f - x_min) / (x_max - x_min or 1) * (width - pad_l - pad_r)

    def sy(a: float) -> float:
        return pad_t + (1 - (a - y_min) / (y_max - y_min or 1)) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" '
             f'aria-label="{dataset} {model} learning curve">']
    # axes
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    for a in (y_min, y_max):
        parts.append(f'<text x="{pad_l-4}" y="{sy(a)+3:.1f}" text-anchor="end" font-size="9" fill="#64748b">{a:.3f}</text>')
    for f in fracs:
        parts.append(f'<text x="{sx(f):.1f}" y="{height-pad_b+12}" text-anchor="middle" font-size="9" fill="#64748b">{int(f*100)}%</text>')
    # lines
    for aug, pts in series.items():
        color = aug_color(aug)
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p['train_fraction']):.1f},{sy(p['accuracy_mean']):.1f}"
                     for i, p in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        for p in pts:
            parts.append(f'<circle cx="{sx(p["train_fraction"]):.1f}" cy="{sy(p["accuracy_mean"]):.1f}" r="2" fill="{color}"/>')
    parts.append("</svg>")
    return "".join(parts)


def subject_curve_svg(curves_by_aug: dict, target: float | None, width: int = 560, height: int = 300) -> str:
    """Wider learning-curve chart for the subject-count study: accuracy vs
    number of subjects, one line per augmentation, with the pre-registered
    target as a horizontal reference line. Self-contained SVG."""
    pad_l, pad_r, pad_t, pad_b = 46, 90, 14, 34
    all_pts = [p for pts in curves_by_aug.values() for p in pts]
    if not all_pts:
        return ""
    ys = [p["accuracy_mean"] for p in all_pts] + ([target] if target else [])
    y_min, y_max = min(ys), max(ys)
    y_min, y_max = y_min - 0.01, y_max + 0.01
    xs = sorted({p["train_fraction"] for p in all_pts})
    x_min, x_max = min(xs), max(xs)

    def sx(x):
        return pad_l + (x - x_min) / (x_max - x_min or 1) * (width - pad_l - pad_r)

    def sy(a):
        return pad_t + (1 - (a - y_min) / (y_max - y_min or 1)) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" aria-label="subject-count learning curve">']
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    for a in (y_min + 0.01, (y_min + y_max) / 2, y_max - 0.01):
        parts.append(f'<text x="{pad_l-5}" y="{sy(a)+3:.1f}" text-anchor="end" font-size="10" fill="#64748b">{a:.2f}</text>')
        parts.append(f'<line x1="{pad_l}" y1="{sy(a):.1f}" x2="{width-pad_r}" y2="{sy(a):.1f}" stroke="#f1f5f9"/>')
    for x in xs:
        parts.append(f'<text x="{sx(x):.1f}" y="{height-pad_b+14}" text-anchor="middle" font-size="10" fill="#64748b">{int(x)}</text>')
    parts.append(f'<text x="{(pad_l+width-pad_r)/2:.0f}" y="{height-4}" text-anchor="middle" font-size="10" fill="#475569">被験者数</text>')
    if target:
        parts.append(f'<line x1="{pad_l}" y1="{sy(target):.1f}" x2="{width-pad_r}" y2="{sy(target):.1f}" stroke="#ef4444" stroke-width="1" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{width-pad_r+4}" y="{sy(target)+3:.1f}" font-size="10" fill="#ef4444">目標 {target}</text>')
    for aug, pts in curves_by_aug.items():
        color = aug_color(aug)
        pts = sorted(pts, key=lambda p: p["train_fraction"])
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p['train_fraction']):.1f},{sy(p['accuracy_mean']):.1f}"
                     for i, p in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{2 if aug == "none" else 1.5}" '
                     f'{"" if aug == "none" else "opacity=0.85"}/>')
        for p in pts:
            parts.append(f'<circle cx="{sx(p["train_fraction"]):.1f}" cy="{sy(p["accuracy_mean"]):.1f}" r="2.5" fill="{color}"/>')
        last = pts[-1]
        parts.append(f'<text x="{width-pad_r+4}" y="{sy(last["accuracy_mean"])+3:.1f}" font-size="9" fill="{color}">{aug}</text>')
    parts.append("</svg>")
    return "".join(parts)


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
    # paper tab (academic structure)
    "abstract",
    "introduction",
    "related-work",
    "setup",
    "results",
    "learning-curves",
    "subject-reduction",
    "discussion",
    "limitations",
    "conclusion",
    "references",
    # dashboard tab (non-paper / operational)
    "ops-progress",
    "ops-runs",
    "ops-audit",
    "ops-reproducibility",
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

    # learning-curve charts: one panel per (dataset, model) that has a sweep
    curves = results.get("learning_curves", {})
    curve_keys = sorted({(k.split("|")[0], k.split("|")[1]) for k in curves})
    curve_panels = []
    for dataset, model in curve_keys:
        # only render panels that actually have multiple fractions (a real sweep)
        sample = curves.get(f"{dataset}|{model}|none", [])
        if len(sample) >= 2:
            svg = learning_curve_svg(dataset, model, curves)
            if svg:
                curve_panels.append({"dataset": dataset, "model": model, "svg": svg})

    # subject-count reduction (Phase 5) + its learning-curve chart
    reduction = results.get("reduction")
    reduction_svg = ""
    if reduction and reduction.get("methods"):
        curves_by_aug = {m["augmentation"]: [{"train_fraction": p["subject_count"],
                                              "accuracy_mean": p["mean"]} for p in m["curve"]]
                         for m in reduction["methods"]}
        reduction_svg = subject_curve_svg(curves_by_aug, reduction.get("target_value"))

    from signal_aug.evaluation.stats import holm_bonferroni

    stats = results.get("stats", [])
    # significance after Holm-Bonferroni step-down over the family of tests
    stats_sorted = holm_bonferroni(stats, alpha=0.05)

    # main table shows full-training-set rows only; the fraction sweep lives in
    # the learning-curve figures, so 858 rows don't flood the table
    summary_full = [s for s in summary_main if s.get("train_fraction", 1.0) == 1.0]

    # headline numbers for the abstract/intro (all derived from data, not typed)
    study_runs = [r for r in results["runs"] if r["status"] == "completed" and r.get("dataset") != "synthetic"]
    dtw_reduction = next(
        (m["reduction_rate"] for m in (reduction or {}).get("methods", []) if m["augmentation"] == "dtw"),
        None,
    )
    facts = {
        "n_study_runs": len(study_runs),
        "n_study_datasets": len({r["dataset"] for r in study_runs}),
        "n_significant": sum(1 for s in stats_sorted if s.get("significant_holm")),
        "target_metric": (reduction or {}).get("target_metric"),
        "target_value": (reduction or {}).get("target_value"),
        "n_star_none": (reduction or {}).get("n_star_none"),
        "dtw_reduction_pct": round(dtw_reduction * 100, 1) if dtw_reduction is not None else None,
    }

    return {
        "facts": facts,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "results": results,
        "summary": summary_full,
        "best_improvements": best_improvements,
        "worst_degradations": worst_degradations,
        "findings": findings_data.get("findings", []),
        "ref": references_index,
        "curve_panels": curve_panels,
        "stats": stats_sorted,
        "aug_colors": AUG_COLORS,
        "reduction": reduction,
        "reduction_svg": reduction_svg,
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
