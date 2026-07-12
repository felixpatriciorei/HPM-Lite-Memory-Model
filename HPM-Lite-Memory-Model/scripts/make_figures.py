"""Canonical figure-generation entry point for HPM-Lite.

This script intentionally delegates to the research-grade reset/generation path
so readers do not have to choose among older make_*_figures.py scripts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    target = repo / "scripts" / "reset_research_grade_figures.py"
    if not target.exists():
        target = repo / "scripts" / "make_research_grade_figures.py"
    if not target.exists():
        raise FileNotFoundError("No research-grade figure generation script found.")
    return subprocess.call([sys.executable, str(target)], cwd=repo)


if __name__ == "__main__":
    raise SystemExit(main())
