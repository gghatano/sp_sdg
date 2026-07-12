#!/usr/bin/env python
"""Validate that required artifacts/ files exist and task_queue.yaml is well-formed."""

from pathlib import Path

import yaml

REQUIRED = [
    "artifacts/state.md",
    "artifacts/task_queue.yaml",
    "artifacts/decision_log.md",
    "artifacts/reproduction_notes.md",
    "artifacts/deviations.md",
    "artifacts/audit_checklist.md",
    "artifacts/limitations.md",
    "artifacts/findings.json",
]

VALID_TASK_STATUSES = {"todo", "in_progress", "done", "blocked"}


def main() -> None:
    problems = [f"missing: {p}" for p in REQUIRED if not Path(p).exists()]
    queue_path = Path("artifacts/task_queue.yaml")
    if queue_path.exists():
        queue = yaml.safe_load(queue_path.read_text())
        for task in queue.get("tasks", []):
            if not {"id", "title", "status", "phase"} <= set(task):
                problems.append(f"task missing keys: {task}")
            elif task["status"] not in VALID_TASK_STATUSES:
                problems.append(f"task {task['id']}: invalid status {task['status']!r}")
    for p in problems:
        print(f"[problem] {p}")
    if problems:
        raise SystemExit(1)
    print("[ok] artifacts are valid")


if __name__ == "__main__":
    main()
