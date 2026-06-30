# What makes the figures look more like research figures

The current plots are honest, but they are still *pilot-study figures*. They feel simple because the dataset behind them is still small.

## To make them truly research-grade, add:

1. **Multiple seeds**  
   Plot mean ± standard deviation or 95% confidence intervals.

2. **Ablation figure**  
   Compare:
   - local baseline
   - full HPM-Lite
   - no episodic
   - no recurrent
   - no router
   - no null slot

3. **Efficiency figure**
   A table or plot with:
   - params
   - VRAM peak
   - tokens/sec
   - wall-clock time

4. **Writer diagnostics**
   Plot:
   - writer recall
   - false write rate
   - missed fact rate
   - retrieval top1
   against distance.

5. **Control figure**
   Show:
   - normal setting
   - shuffled values
   - random writes
   - no retrieval
   - missing-key query

6. **Sample-efficiency curve**
   Exact accuracy versus training step for local vs HPM-Lite.

## Important truth

A figure looks “research-grade” mostly because:
- the experiment design is strong,
- the statistics are complete,
- the caption is precise,
- and the comparisons answer a real question.

Fancy styling helps, but stronger data helps more.
