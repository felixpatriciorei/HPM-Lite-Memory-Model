# Research figure design notes

These figure-generation choices are based on conservative scientific-visualization practice:

- Show seed-level raw points, not only aggregate bars.
- State what error bars mean; use sample standard deviation for seed sweeps unless a stronger statistical claim is justified.
- Use compact figures with one clear message per panel.
- Export vector formats (`.svg`, `.pdf`) as well as `.png`.
- Treat timing/VRAM comparisons as approximate when hardware differs.
- Keep local-baseline writer metrics out of mechanism claims.

For HPM-Lite specifically, the most important diagnostic plots are not decorative. They should answer these questions:

1. Did exact answer accuracy improve across seeds?
2. Was retrieval itself correct?
3. Did errors come from writer misses/false writes?
4. Did training show instability or drift?
5. What was the cost in parameters, VRAM, and wall time?

The advanced figure script writes an audit file beside the generated figures so the README can link to the assumptions behind the charts.
