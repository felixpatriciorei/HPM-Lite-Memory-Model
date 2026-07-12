"""Run both figure systems: research-grade core figures and advanced atlas.

This is intentionally separate from scripts/make_figures.py so the canonical
README figure command stays simple. Use this when you want the full graph/stat
refresh across all committed research data:

    python scripts/make_all_research_graphs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(repo: Path, script_name: str) -> int:
    target = repo / "scripts" / script_name
    if not target.exists():
        print(f"skip missing {target}")
        return 0
    print(f"running {target.relative_to(repo)}")
    return subprocess.call([sys.executable, str(target)], cwd=repo)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    rc = run(repo, "reset_research_grade_figures.py")
    if rc != 0:
        return rc
    return run(repo, "make_advanced_research_atlas.py")


if __name__ == "__main__":
    raise SystemExit(main())
