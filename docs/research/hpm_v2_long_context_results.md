# HPM-Lite v2 long-context results

This document summarizes the merged Kaggle + PC overnight matrix. The main headline uses canonical Kaggle rows because those rows share the same runtime/source and batch policy. PC rows are kept as supplemental connected-compute evidence.

## Canonical Kaggle summary

| Seq len | Writer schedule | n | Exact mean ± SD | Writer true-fact mean | Retrieval top-1 mean |
|---:|---:|---:|---:|---:|---:|
| 4096 | tf200 | 4 | 0.8000 ± 0.1472 | 0.8539 | 1.0000 |
| 4096 | tf600 | 8 | 0.9938 ± 0.0116 | 0.9914 | 1.0000 |
| 8192 | tf200 | 6 | 0.7250 ± 0.1440 | 0.7344 | 0.9907 |
| 8192 | tf600 | 8 | 0.9750 ± 0.0378 | 0.9766 | 1.0000 |
| 12288 | tf200 | 2 | 0.6500 ± 0.0707 | 0.7437 | 1.0000 |
| 12288 | tf600 | 2 | 0.9500 ± 0.0707 | 0.9438 | 1.0000 |

## Writer-schedule gaps

| Seq len | tf600 - tf200 exact gap | tf600 - tf200 writer gap |
|---:|---:|---:|
| 4096 | +19.37 pp | +13.75 pp |
| 8192 | +25.00 pp | +24.22 pp |
| 12288 | +30.00 pp | +20.00 pp |

## Main interpretation

- Full-run writer supervision (`tf600`) remains strong from 4096 through 12288 tokens in the current synthetic KV benchmark.
- Early-stop writer supervision (`tf200`) is a negative control: exact accuracy drops sharply as the context gets longer.
- Retrieval top-1 stays at or near 1.0 in nearly all groups, so the dominant failure mode is still writing/missed facts, not retrieval collapse.
- PC rows are real connected-compute contributions, but README headline claims should use canonical Kaggle rows unless hardware/batch effects are explicitly discussed.

## Produced files

- `results/processed/hpm_v2_long_context_matrix.csv`
- `results/processed/hpm_v2_long_context_summary.csv`
- `results/processed/hpm_v2_writer_schedule_gaps.csv`
- `results/processed/hpm_v2_training_dynamics_long.csv`
- `results/figures/advanced/long_context/`