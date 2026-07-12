# HPM-Lite paper figure design audit

This replaces the earlier quick figure approach. The goal is not to make plots look busy; the goal is to make every figure defensible.

## What changed

1. **No bar-only result plots.** Main quantitative panels show raw seed points plus mean ± sample SD.
2. **No hidden uncertainty.** Error bars are explicitly sample standard deviation, not SEM and not a claimed confidence interval.
3. **No misleading local memory diagnostics.** Local baseline writer/memory columns are treated as bookkeeping noise and are not interpreted.
4. **Matched-seed comparison is separated from extra seeds.** The main 2048 comparison uses HPM seeds 0–2 versus local seeds 0–2. Extra local seed 3 is shown only in a supplemental panel.
5. **Wall time is not overclaimed.** HPM was run locally and local baselines were run on Kaggle T4s, so speed panels are context only, not a hardware-fair architecture claim.
6. **Vector outputs are included.** Each figure is saved as PNG, SVG, and PDF.
7. **Architecture and task are separated from numeric results.** A schematic explains the mechanism; result figures show data.

## Generated outputs

Run:

```cmd
"%VENV%\Scripts\python.exe" scripts\make_research_figures.py
```

Outputs are written to:

```text
results\figures\paper\
```

Expected files:

```text
fig_01_model_task_schematic.{png,svg,pdf}
fig_02_main_2048_results.{png,svg,pdf}
fig_03_writer_retrieval_diagnostics.{png,svg,pdf}
fig_04_hpm_training_dynamics.{png,svg,pdf}   # only if step logs are available
fig_05_supplemental_seed_checks.{png,svg,pdf}
paper_results_table.csv
figure_manifest.csv
figure_audit_report.md
```

## Caption guidance

### Figure 1
HPM-Lite architecture and synthetic KV-memory task. The model routes between local mixer state, recurrent state, and episodic retrieval. FACT tokens are written by a learned writer, QUERY tokens retrieve from episodic memory, and the answer head predicts the target value.

### Figure 2
Main 2048-token KV-memory comparison. HPM-Lite learned writer is compared against a fixed-window local Transformer baseline using matched seeds. Points are individual seeds; horizontal markers and error bars show mean ± sample SD. The speed/resource panel is context only because hardware differs between local and Kaggle runs.

### Figure 3
HPM learned-writer diagnostics. Retrieval top-1 remains perfect in these runs, while remaining answer errors align with nonzero missed fact rates. This should be written as a diagnostic observation, not a formal causal proof.

### Figure 4
Training dynamics from step logs. These curves show learning stability and seed variability across training; CE and loss use log1p scaling to keep spikes readable.

### Figure 5
Supplemental seed checks. Shows all available local baseline seeds and optional 512→2048 HPM scaling if the 512 processed sweep exists.

## Rules for writing about these figures

Use:

> At 2048 tokens with a 256-token local window, HPM-Lite learned writer achieved 98.33% mean exact accuracy over three seeds, while the matched local Transformer baseline achieved 0.00%. Error bars show sample SD over seeds.

Do not use:

> HPM-Lite proves general long-context intelligence.

Do not use:

> HPM-Lite is faster than the local Transformer.

The current data supports a controlled synthetic KV-memory claim, not a general language-modeling or hardware-speed claim.
