# Figure Caption Templates

## Figure 1 — Main result

**Figure 1. Long-range key-value recall under fixed local window.**
A fixed-window local Transformer fails once the required fact is outside the attention window, while HPM-Lite maintains exact recall through episodic memory retrieval. Points show mean over seeds; shaded regions show standard deviation. Panel D reports the compute/accuracy frontier, with marker size proportional to peak VRAM.

## Figure 2 — Ablation heatmap

**Figure 2. Component ablations identify episodic memory as the decisive path for long-range exact recall.**
Rows remove or corrupt one component at a time. Columns are sequence lengths. Color indicates exact answer accuracy. The expected failure of no-retrieval, shuffled-value, and no-episodic controls supports that HPM-Lite is using memory rather than a dataset shortcut.

## Figure 3 — Learned writer diagnostics

**Figure 3. Supervised learned writing separates memory allocation from retrieval.**
Writer recall, missed fact rate, false write rate, and retrieval top1 are reported over distance. High answer accuracy requires both high fact-write recall and correct retrieval.

## Figure 4 — Retrieval score separation

**Figure 4. Retrieval scores separate correct memory slots from distractors.**
Correct-slot, best-wrong-slot, and null-slot score distributions are shown over evaluation queries. Larger margins indicate more reliable content-addressable retrieval.

## Figure 5 — Router path usage

**Figure 5. Router weights reveal path specialization.**
Each point/token is represented by its local, recurrent, and episodic router weights. Fact/query/answer/noise tokens are shown separately to test whether the router uses episodic memory specifically where exact recall is needed.

## Figure 6 — Memory interaction graph

**Figure 6. Case-study memory graph for a single long-range query.**
Nodes are fact tokens, memory slots, query tokens, and answer tokens. Edges represent write and retrieval decisions; edge thickness is proportional to write probability or retrieval score.
