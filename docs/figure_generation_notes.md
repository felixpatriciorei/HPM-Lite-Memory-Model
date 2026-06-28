# Research figure generation

Run from the repository root after creating:

- `results/processed/learned_writer_2048_seed_sweep.csv`
- `results/processed/local_2048_seed_sweep.csv`

Command:

```cmd
"%VENV%\Scripts\python.exe" scripts\make_research_figures.py
```

Expected outputs:

- `results/figures/fig_01_architecture_memory_flow.png`
- `results/figures/fig_02_2048_accuracy_mean_std.png`
- `results/figures/fig_03_2048_answer_ce_mean_std.png`
- `results/figures/fig_04_2048_per_seed_accuracy.png`
- `results/figures/fig_05_2048_writer_diagnostics.png`
- `results/figures/fig_06_2048_parameter_count.png`
- `results/figures/fig_07_2048_hpm_learning_curves.png` if step logs are available
- `results/figures/fig_08_2048_writer_learning_curves.png` if writer columns are available in step logs
- `results/figures/summary_2048_table.csv`
- `results/figures/figure_manifest.csv`

The main comparison uses HPM's three seeds and the matched first three local baseline seeds.
The fourth local baseline seed is preserved in the summary table as extra evidence, but it is not used for the matched main figure.
