# Repository front-page audit

## Previous problem

The earlier README summarized the result but did not properly surface the figures, diagnostics, and figure-generation workflow. That made the repo look weaker than the actual experiment.

## Fix in this overlay

The README now treats the generated paper figures as first-class artifacts:

- `fig_02_main_2048_results.png` is the hero result.
- `fig_01_model_task_schematic.png` explains the mechanism.
- `fig_03_writer_retrieval_diagnostics.png` explains why remaining HPM errors are writer-related.
- `fig_04_hpm_training_dynamics.png` shows training behavior instead of only final metrics.
- `fig_05_supplemental_seed_checks.png` exposes extra seed/scaling checks.

## Remaining limitations

The README is still only as strong as the committed files. Before pushing:

1. Generate figures.
2. Verify images exist.
3. Verify processed CSVs exist.
4. Check GitHub rendering after push.

## Do not hide these caveats

- The task is synthetic.
- The comparison is not parameter-matched yet.
- Main HPM result uses 3 seeds.
- Local baseline writer columns are not meaningful.
- Speed comparisons are hardware-mixed and should be treated cautiously.
