#!/usr/bin/env python3
"""Delete legacy HPM-Lite figure outputs and regenerate the research-grade suite."""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIRS = [
    ROOT / "results" / "figures" / "paper",
    ROOT / "results" / "figures" / "advanced",
    ROOT / "docs" / "figures",
]
LEGACY_FILES = [
    ROOT / "results" / "figures" / "figure_manifest.csv",
    ROOT / "results" / "figures" / "figure_audit_report.md",
]

def main() -> int:
    print("HPM-Lite full graph reset")
    print("Deleting legacy figure directories...")
    for path in LEGACY_DIRS:
        if path.exists():
            print(f"  delete {path.relative_to(ROOT)}")
            shutil.rmtree(path)
    for path in LEGACY_FILES:
        if path.exists():
            print(f"  delete {path.relative_to(ROOT)}")
            path.unlink()
    print("Generating research-grade figures...")
    cmd = [sys.executable, str(ROOT / "scripts" / "make_research_grade_figures.py")]
    return subprocess.call(cmd, cwd=str(ROOT))

if __name__ == "__main__":
    raise SystemExit(main())
