#!/usr/bin/env python
"""Compact run logs: keep logs for failed runs, truncate logs of completed runs
older than the newest N (default 50). Manifests and metrics are never touched.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=int, default=50)
    args = parser.parse_args()

    manifests = sorted(Path("runs/manifests").glob("*.json"), key=lambda p: p.stat().st_mtime)
    completed = [json.loads(p.read_text(encoding="utf-8")) for p in manifests]
    completed = [m for m in completed if m.get("status") == "completed" and m.get("log_path")]
    n_truncated = 0
    for manifest in completed[: -args.keep] if args.keep else completed:
        log = Path(manifest["log_path"])
        if log.exists() and log.stat().st_size > 200:
            lines = log.read_text(encoding="utf-8").splitlines()
            log.write_text(
                "\n".join(lines[:2] + ["... [truncated by compact_logs.py]"] + lines[-2:]) + "\n",
                encoding="utf-8",
            )
            n_truncated += 1
    print(f"[compact] truncated {n_truncated} completed-run logs (kept newest {args.keep})")


if __name__ == "__main__":
    main()
