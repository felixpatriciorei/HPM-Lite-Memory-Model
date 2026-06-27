from pathlib import Path
import csv, math
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

with (RESULTS / "oracle_distance_results.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: int(r["distance"]))

dist = [int(r["distance"]) for r in rows]
local_exact = [float(r["local_exact"]) for r in rows]
hpm_exact = [float(r["hpm_exact"]) for r in rows]
local_ce = [float(r["local_ce"]) for r in rows]
hpm_ce = [float(r["hpm_ce"]) for r in rows]

def save(name):
    plt.tight_layout()
    plt.savefig(FIGS / name, dpi=220, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(8.5,5))
plt.plot(dist, local_exact, marker="o", label="Local Transformer")
plt.plot(dist, hpm_exact, marker="o", label="HPM-Lite")
plt.xscale("log", base=2); plt.xticks(dist, [str(d) for d in dist])
plt.ylim(-0.05,1.05); plt.xlabel("Sequence length"); plt.ylabel("Exact answer accuracy")
plt.title("Exact recall under increasing fact-to-query distance")
plt.grid(True, alpha=.35); plt.legend(); save("01_exact_recall_vs_distance.png")

plt.figure(figsize=(8.5,5))
plt.plot(dist, local_ce, marker="o", label="Local Transformer")
plt.plot(dist, hpm_ce, marker="o", label="HPM-Lite")
plt.xscale("log", base=2); plt.xticks(dist, [str(d) for d in dist])
plt.xlabel("Sequence length"); plt.ylabel("Answer-position cross entropy")
plt.title("Answer CE grows for local attention, stays near zero for HPM-Lite")
plt.grid(True, alpha=.35); plt.legend(); save("02_answer_ce_vs_distance.png")

plt.figure(figsize=(8.5,5))
plt.plot(dist, [math.exp(-x) for x in local_ce], marker="o", label="Local Transformer")
plt.plot(dist, [math.exp(-x) for x in hpm_ce], marker="o", label="HPM-Lite")
plt.xscale("log", base=2); plt.yscale("log"); plt.xticks(dist, [str(d) for d in dist])
plt.xlabel("Sequence length"); plt.ylabel("p(correct answer) implied by CE")
plt.title("Correct-answer probability from exp(-CE)")
plt.grid(True, alpha=.35); plt.legend(); save("03_correct_answer_probability_from_ce.png")

plt.figure(figsize=(8.5,5))
plt.bar([str(d) for d in dist], [a-b for a,b in zip(local_ce,hpm_ce)])
plt.xlabel("Sequence length"); plt.ylabel("CE gap: local minus HPM-Lite")
plt.title("Cross-entropy gap widens with distance")
plt.grid(axis="y", alpha=.35); save("04_ce_gap_by_distance.png")

plt.figure(figsize=(8.5,5))
eps = 1e-4
plt.plot(dist, [max(eps, 1-x) for x in local_exact], marker="o", label="Local Transformer")
plt.plot(dist, [max(eps, 1-x) for x in hpm_exact], marker="o", label="HPM-Lite")
plt.xscale("log", base=2); plt.yscale("log"); plt.xticks(dist, [str(d) for d in dist])
plt.xlabel("Sequence length"); plt.ylabel("Error rate, log scale")
plt.title("Exact-recall error rate")
plt.grid(True, alpha=.35); plt.legend(); save("05_error_rate_log_scale.png")

plt.figure(figsize=(8.5,5))
plt.plot(dist, [float(r["local_params"])/1e6 for r in rows], marker="o", label="Local Transformer")
plt.plot(dist, [float(r["hpm_params"])/1e6 for r in rows], marker="o", label="HPM-Lite")
plt.xscale("log", base=2); plt.xticks(dist, [str(d) for d in dist])
plt.xlabel("Sequence length"); plt.ylabel("Parameters, millions")
plt.title("Parameter counts used in current runs")
plt.grid(True, alpha=.35); plt.legend(); save("06_parameter_count_by_run.png")

with (RESULTS / "learned_writer_results.csv").open(newline="", encoding="utf-8") as f:
    learned = list(csv.DictReader(f))
learned.sort(key=lambda r: int(r["distance"]))
ldist = [int(r["distance"]) for r in learned]

plt.figure(figsize=(8.5,5))
plt.plot(ldist, [float(r["local_exact"]) for r in learned], marker="o", label="Local exact")
plt.plot(ldist, [float(r["hpm_exact"]) for r in learned], marker="o", label="HPM-Lite exact")
plt.plot(ldist, [float(r["writer_recall"]) for r in learned], marker="o", label="Writer recall")
plt.plot(ldist, [float(r["retrieval_top1"]) for r in learned], marker="o", label="Retrieval top1")
plt.xscale("log", base=2); plt.xticks(ldist, [str(d) for d in ldist])
plt.ylim(-0.05,1.05); plt.xlabel("Sequence length"); plt.ylabel("Rate")
plt.title("Learned writer: exact recall, write recall, and retrieval")
plt.grid(True, alpha=.35); plt.legend(); save("07_learned_writer_progress.png")

plt.figure(figsize=(8.5,5))
labels = []
values = []
for r in learned:
    labels += [f'{r["distance"]} local CE', f'{r["distance"]} HPM CE']
    values += [float(r["local_ce"]), float(r["hpm_ce"])]
plt.bar(labels, values)
plt.ylabel("Answer-position cross entropy")
plt.title("Learned-writer CE comparison")
plt.grid(axis="y", alpha=.35); save("08_learned_writer_ce_comparison.png")

print(f"Wrote figures to {FIGS}")
