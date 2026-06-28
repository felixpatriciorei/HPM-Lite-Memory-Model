# README design notes

This README version was rewritten to function as the repository front page, not just a text summary.

## What changed

- The main 2048 result figure is now placed at the top of the README.
- The model/task schematic is included directly, not hidden in `results/figures`.
- Diagnostics and learning curves are linked and embedded.
- The 2048 seed-level table is still visible for people who do not open the figures.
- Error bars are explicitly described as sample standard deviation across seeds.
- The local baseline's writer columns are called out as bookkeeping artifacts.
- The README includes a direct asset checker: `scripts/check_readme_assets.py`.

## Design standard used

The README should answer five questions quickly:

1. What is this project?
2. Why does it matter?
3. What is the strongest result?
4. Can I inspect the figures/data behind it?
5. What should I not overclaim?

The front page now follows that order.

## Important policy

Do not commit this README unless the figure assets exist. Run:

```bash
python scripts/make_research_figures.py
python scripts/check_readme_assets.py
```

If the checker fails, the GitHub front page will have broken images or links.
