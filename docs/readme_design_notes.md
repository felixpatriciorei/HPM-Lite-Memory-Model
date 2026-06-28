# README design notes

This README was rewritten as a public-facing research-engineering front page, not a hype page.

Design choices:

- Lead with the narrow research question.
- State what the project is not.
- Separate v1 headline results from v2 development results.
- Avoid claiming v2 has a full 2048 seed sweep until the CSV exists in the repo.
- Make reproduction commands visible from the front page.
- Link figures and processed CSVs directly.
- Use the resume section as a concise but defensible portfolio description.

The README should be identical on `main` and `hpm-v2-training` after merge. If a branch does not contain the linked files yet, merge the v2 branch before updating README on `main`.

External design references used while drafting:

- GitHub/open-source community guidance emphasizes clear README/contribution documentation as part of a healthy project.
- PLOS figure guidance emphasizes objective data display and avoiding figure designs that obscure the data.
- Nature figure guidance emphasizes defined elements, clear hierarchy, and removing unnecessary decoration.
- NeurIPS-style review checklists require clearly described error bars and experimental uncertainty.
