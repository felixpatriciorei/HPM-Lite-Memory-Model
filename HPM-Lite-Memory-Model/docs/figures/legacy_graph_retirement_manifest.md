# Legacy graph retirement manifest

The research-grade reset replaces the old figure system. Run:

```bash
python scripts/reset_research_grade_figures.py
```

This intentionally deletes the legacy figure directories listed below and regenerates `results/figures/research_grade/`. Git can restore the old files if needed before commit.

## Retired paths

- `results/figures/paper/fig_01_model_task_schematic.pdf`
- `results/figures/paper/fig_01_model_task_schematic.png`
- `results/figures/paper/fig_01_model_task_schematic.svg`
- `results/figures/paper/fig_02_main_2048_results.pdf`
- `results/figures/paper/fig_02_main_2048_results.png`
- `results/figures/paper/fig_02_main_2048_results.svg`
- `results/figures/paper/fig_03_writer_retrieval_diagnostics.pdf`
- `results/figures/paper/fig_03_writer_retrieval_diagnostics.png`
- `results/figures/paper/fig_03_writer_retrieval_diagnostics.svg`
- `results/figures/paper/fig_04_hpm_training_dynamics.pdf`
- `results/figures/paper/fig_04_hpm_training_dynamics.png`
- `results/figures/paper/fig_04_hpm_training_dynamics.svg`
- `results/figures/paper/fig_05_supplemental_seed_checks.pdf`
- `results/figures/paper/fig_05_supplemental_seed_checks.png`
- `results/figures/paper/fig_05_supplemental_seed_checks.svg`
- `results/figures/paper/figure_audit_report.md`
- `results/figures/paper/figure_manifest.csv`
- `results/figures/paper/paper_results_table.csv`
- `results/figures/advanced/advanced_figure_audit.md`
- `results/figures/advanced/advanced_figure_manifest.csv`
- `results/figures/advanced/fig_adv_01_exact_raw_seed_points.pdf`
- `results/figures/advanced/fig_adv_01_exact_raw_seed_points.png`
- `results/figures/advanced/fig_adv_01_exact_raw_seed_points.svg`
- `results/figures/advanced/fig_adv_02_writer_error_decomposition.pdf`
- `results/figures/advanced/fig_adv_02_writer_error_decomposition.png`
- `results/figures/advanced/fig_adv_02_writer_error_decomposition.svg`
- `results/figures/advanced/fig_adv_03_writer_success_vs_exact.pdf`
- `results/figures/advanced/fig_adv_03_writer_success_vs_exact.png`
- `results/figures/advanced/fig_adv_03_writer_success_vs_exact.svg`
- `results/figures/advanced/fig_adv_04_efficiency_frontier.pdf`
- `results/figures/advanced/fig_adv_04_efficiency_frontier.png`
- `results/figures/advanced/fig_adv_04_efficiency_frontier.svg`
- `results/figures/advanced/fig_adv_05_training_dynamics.pdf`
- `results/figures/advanced/fig_adv_05_training_dynamics.png`
- `results/figures/advanced/fig_adv_05_training_dynamics.svg`
- `results/figures/advanced/long_context/fig_long_01_exact_by_length_schedule.pdf`
- `results/figures/advanced/long_context/fig_long_01_exact_by_length_schedule.png`
- `results/figures/advanced/long_context/fig_long_01_exact_by_length_schedule.svg`
- `results/figures/advanced/long_context/fig_long_02_writer_rate_vs_exact.pdf`
- `results/figures/advanced/long_context/fig_long_02_writer_rate_vs_exact.png`
- `results/figures/advanced/long_context/fig_long_02_writer_rate_vs_exact.svg`
- `results/figures/advanced/long_context/fig_long_03_schedule_gap.pdf`
- `results/figures/advanced/long_context/fig_long_03_schedule_gap.png`
- `results/figures/advanced/long_context/fig_long_03_schedule_gap.svg`
- `results/figures/advanced/long_context/fig_long_04_exact_writer_retrieval_means.pdf`
- `results/figures/advanced/long_context/fig_long_04_exact_writer_retrieval_means.png`
- `results/figures/advanced/long_context/fig_long_04_exact_writer_retrieval_means.svg`
- `results/figures/advanced/long_context/fig_long_05_training_dynamics_exact.pdf`
- `results/figures/advanced/long_context/fig_long_05_training_dynamics_exact.png`
- `results/figures/advanced/long_context/fig_long_05_training_dynamics_exact.svg`
- `results/figures/advanced/long_context/fig_long_06_answer_ce_distribution.pdf`
- `results/figures/advanced/long_context/fig_long_06_answer_ce_distribution.png`
- `results/figures/advanced/long_context/fig_long_06_answer_ce_distribution.svg`
- `results/figures/advanced/long_context/fig_long_07_peak_vram_scaling.pdf`
- `results/figures/advanced/long_context/fig_long_07_peak_vram_scaling.png`
- `results/figures/advanced/long_context/fig_long_07_peak_vram_scaling.svg`
- `results/figures/advanced/long_context/long_context_figure_audit.md`
- `results/figures/advanced/long_context/long_context_figure_manifest.csv`
- `docs/figures/01_exact_recall_vs_distance.png`
- `docs/figures/02_answer_ce_vs_distance.png`
- `docs/figures/03_correct_answer_probability_from_ce.png`
- `docs/figures/04_ce_gap_by_distance.png`
- `docs/figures/05_error_rate_log_scale.png`
- `docs/figures/06_parameter_count_by_run.png`
- `docs/figures/07_learned_writer_progress.png`
- `docs/figures/08_learned_writer_ce_comparison.png`
- `docs/figures/answer_ce_vs_distance.png`
- `docs/figures/exact_gain_by_distance.png`
- `docs/figures/exact_recall_vs_distance.png`
- `docs/figures/hpm_lite_model_paths.png`
