#!/usr/bin/env python
"""Build report/dist/index.html from aggregated results."""

from signal_aug.reporting.build import build_report


def main() -> None:
    out = build_report()
    print(f"[report] built {out}")


if __name__ == "__main__":
    main()
