#!/usr/bin/env python3
"""Legacy entrypoint. The HPM-Lite figure system now delegates to the research-grade pipeline."""
from pathlib import Path
import runpy
ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts" / "make_research_grade_figures.py"), run_name="__main__")
