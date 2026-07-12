# Migrating HPM-Lite-2 work back into HPM-Lite

Goal: keep one public repository: `HPM-Lite-Memory-Model`.

Recommended safe path:

```cmd
git remote -v
git status
"%VENV%\Scripts\python.exe" -m pytest -q
```

If this local checkout now points to the HPM-Lite repo:

```cmd
git push origin hpm-v2-training
```

Then merge into `main` locally:

```cmd
git checkout main
git pull origin main
git merge hpm-v2-training --no-ff
"%VENV%\Scripts\python.exe" -m pytest -q
"%VENV%\Scripts\python.exe" scripts\make_advanced_research_figures.py
git add scripts\make_advanced_research_figures.py docs\research_figure_design_notes.md docs\hpm_lite_repo_migration.md results\figures\advanced results\processed\advanced_research_stats.csv
git commit -m "Add HPM-Lite v2 advanced analysis figures"
git push origin main
```

Only delete `HPM-Lite-2` after verifying on GitHub that `HPM-Lite-Memory-Model` contains the v2 files, tests, processed CSVs, and advanced figures.
