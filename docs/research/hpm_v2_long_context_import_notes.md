# Long-context import notes

This overlay merges three uploaded result archives:

- `hpm_v2_connected_compute_matrix.zip`
- `hpm_v2_long_overnight_matrix.zip`
- `pc_long_overnight_matrix.zip`

Deduplication rule:

- If the same raw CSV filename appeared in both Kaggle archives, the later `long` archive version was kept.
- PC rows were kept when they had distinct `_pc.csv` filenames and completed CSV rows.
- Failed/empty PC attempts were not included in the processed matrix.

Claiming rule:

- Use `canonical_kaggle` rows for headline statistics.
- Use `all_workers` rows to show the connected-compute contribution, but mark them as mixed hardware/batch evidence.

Apply this overlay from the repo root with:

```cmd
tar -xf C:\Users\win10\Downloads\hpm_long_context_research_overlay.zip
```

Then verify:

```cmd
python -m pytest -q
python scripts\make_long_context_research_figures.py
python scripts\make_advanced_research_figures.py
git status
```
