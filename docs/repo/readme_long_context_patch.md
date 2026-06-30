## Long-context v2 stress matrix

The overnight long-context matrix expands HPM-Lite v2 beyond the earlier 512/2048-token runs. The clean headline rows below use the canonical Kaggle runs only, so the statistics are not mixed with different PC hardware or batch-size settings.

| Seq len | Writer schedule | Seeds | Exact accuracy | Writer true-fact rate | Retrieval top-1 |
|---:|---:|---:|---:|---:|---:|
| 4096 | tf200 | 4 | 0.8000 ± 0.1472 | 0.8539 | 1.0000 |
| 4096 | tf600 | 8 | 0.9938 ± 0.0116 | 0.9914 | 1.0000 |
| 8192 | tf200 | 6 | 0.7250 ± 0.1440 | 0.7344 | 0.9907 |
| 8192 | tf600 | 8 | 0.9750 ± 0.0378 | 0.9766 | 1.0000 |
| 12288 | tf200 | 2 | 0.6500 ± 0.0707 | 0.7437 | 1.0000 |
| 12288 | tf600 | 2 | 0.9500 ± 0.0707 | 0.9438 | 1.0000 |


The main result is not just that exact accuracy remains high with full writer supervision. The stronger finding is the schedule contrast: when writer teacher forcing stops early at step 200, exact recall falls hard even though retrieval top-1 remains near perfect. That supports the current diagnosis that long-context v2 failures are dominated by write quality, not retrieval collapse.

![Long-context exact accuracy by writer schedule](results/figures/advanced/long_context/fig_long_01_exact_by_length_schedule.png)

![Writer rate versus exact recall](results/figures/advanced/long_context/fig_long_02_writer_rate_vs_exact.png)

![Writer schedule gap](results/figures/advanced/long_context/fig_long_03_schedule_gap.png)

More figures are in `results/figures/advanced/long_context/`, and the merged matrix is in `results/processed/hpm_v2_long_context_matrix.csv`.
