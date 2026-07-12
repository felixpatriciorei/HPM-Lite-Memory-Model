# Branch sync protocol

Goal: make `main` and `hpm-v2-training` share the same README and v2 files.

Recommended sequence:

```cmd
git checkout hpm-v2-training
git status
python -m pytest -q
python scripts\make_advanced_research_figures.py
git add README.md docs\readme_design_notes.md docs\hpm_lite_branch_sync.md docs\research_figure_design_notes.md docs\hpm_lite_repo_migration.md scripts\make_advanced_research_figures.py results\figures\advanced results\processed\advanced_research_stats.csv
git commit -m "Update README and advanced research figures"
git push origin hpm-v2-training

git checkout main
git pull origin main
git merge hpm-v2-training --no-ff
python -m pytest -q
python scripts\make_advanced_research_figures.py
git push origin main
```

Only delete the old HPM-Lite-2 repository after GitHub `main` contains:

- `hpm_lite/hpm_v2.py`
- `hpm_lite/hpm_v2_model.py`
- `results/processed/hpm_v2_512_seed_sweep.csv`
- `results/processed/hpm_v2_2048_tf600_lw03_seed_sweep.csv`
- `results/figures/advanced/`
