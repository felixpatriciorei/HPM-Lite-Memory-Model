# HPM-Lite Research-Grade Figure Plan

## Goal

Replace simple pilot plots with a figure suite that proves mechanism, not just endpoint accuracy.

The project should not try to make the plots look complex for decoration. A research-grade figure is complex because it shows:
- uncertainty
- ablations
- controls
- internal mechanism
- compute cost
- failure behavior

## Current claim

A compact HPM-style model can preserve and retrieve long-range key-value facts that a fixed-window local Transformer cannot access.

## Current evidence level

Strong pilot evidence:
- oracle/null-slot memory solves 512, 2048, 4096, 8192 synthetic KV recall.
- supervised learned writer solves 512 and 2048 synthetic KV recall.
- local baseline fails when the fact is outside the local window.

Not yet paper-grade:
- mostly single seed
- no full ablation suite
- no per-step logs
- no router traces
- no retrieval score distributions
- no memory interaction graph traces
- incomplete VRAM/tokens/sec logging

---

# Figure 1 — Main Result Multipanel

Filename:
`docs/figures/fig1_main_result_multipanel.png`

Panels:
A. Exact answer accuracy vs sequence length  
B. Answer-position CE vs sequence length  
C. Correct-answer probability, exp(-CE), log scale  
D. Efficiency frontier: exact accuracy vs tokens/sec, marker size = peak VRAM

Needed columns:
`run_id, commit, model, write_mode, seq_len, window, seed, exact, ce, tokens_per_sec, peak_vram_mb, params`

Why it matters:
This is the headline figure. It shows the main claim and forces compute reporting.

Minimum data:
- local baseline: 512, 2048, 4096, 8192
- HPM-Lite oracle/null-slot: 512, 2048, 4096, 8192
- HPM-Lite learned writer: 512, 2048
- 3 seeds where possible

---

# Figure 2 — Ablation Heatmap

Filename:
`docs/figures/fig2_ablation_heatmap.png`

Rows:
- local
- hpm_lite_full
- hpm_lite_no_episodic
- hpm_lite_no_recurrent
- hpm_lite_no_router
- hpm_lite_no_null_slot
- hpm_lite_random_write
- hpm_lite_shuffled_values

Columns:
- 512
- 2048
- 4096
- 8192

Cell value:
- exact answer accuracy, with an optional separate CE heatmap

Needed columns:
`run_id, model_variant, seq_len, seed, exact, ce, retrieval_top1, writer_recall`

Why it matters:
This is the most important research-grade figure. It proves which component is doing the work.

Expected pattern:
- full HPM works
- local fails
- no-episodic fails on long-range recall
- shuffled-memory fails
- random-write fails
- no-null-slot should be worse on missing/no-match tests

---

# Figure 3 — Learned Writer Diagnostics

Filename:
`docs/figures/fig3_learned_writer_diagnostics.png`

Panels:
A. writer recall vs sequence length  
B. missed fact rate vs sequence length  
C. false write rate vs sequence length  
D. retrieval top1 vs sequence length

Needed columns:
`run_id, seq_len, seed, writer_recall, missed_fact_rate, false_write_rate, retrieval_top1, avg_written_slots`

Why it matters:
This shows the model is moving away from oracle writes. It decomposes answer success into writing success and retrieval success.

Current available points:
- 512 learned writer
- 2048 learned writer

Missing:
- 4096 learned writer
- 8192 learned writer
- multiple seeds

---

# Figure 4 — Retrieval Score Separation

Filename:
`docs/figures/fig4_retrieval_score_distribution.png`

Possible plot types:
- violin plot
- ridge plot
- histogram
- boxen plot

Groups:
- correct slot score
- best wrong slot score
- null slot score
- retrieval margin

Needed per-example columns:
`run_id, example_id, seq_len, seed, correct_slot_score, best_wrong_slot_score, null_slot_score, retrieval_margin, correct_slot_rank, answer_correct`

Why it matters:
This shows whether memory retrieval is truly separating correct facts from distractors.

---

# Figure 5 — Router Ternary / Path-Usage Plot

Filename:
`docs/figures/fig5_router_path_usage.png`

Data:
- alpha_local
- alpha_recurrent
- alpha_episodic
- token_type: noise, fact, query, answer
- position bucket
- answer_correct

Plot options:
- ternary scatter
- stacked path-usage bars by token type
- router entropy curve over sequence position

Needed columns:
`run_id, example_id, position, token_type, alpha_local, alpha_recurrent, alpha_episodic, router_entropy, answer_correct`

Why it matters:
This is the most HPM-specific mechanism figure. It can show path specialization:
- local path for nearby tokens
- episodic path for queries/answers
- recurrent path for continuity

---

# Figure 6 — Memory Interaction Graph

Filename:
`docs/figures/fig6_memory_interaction_graph_case_study.png`

Graph design:
- nodes: fact tokens, query token, answer token, memory slots
- write edges: fact token -> memory slot
- retrieval edges: query token -> memory slot
- answer edge: retrieved slot -> answer
- edge thickness: write probability or retrieval score
- edge style: correct vs distractor

Needed trace data:
`run_id, example_id, node_id, node_type, label, position`
`run_id, example_id, src_node, dst_node, edge_type, score, is_correct`

Why it matters:
This gives the graph/network-style visual the project currently lacks. It is not decorative; it is a mechanistic case study.

---

# Figure 7 — Failure and Control Matrix

Filename:
`docs/figures/fig7_failure_controls.png`

Conditions:
- normal
- no retrieval
- shuffled values
- random keys
- random writes
- missing key
- near-duplicate keys

Metrics:
- exact accuracy
- CE
- retrieval top1
- null slot selection
- writer false write rate

Needed columns:
`run_id, control, model, seq_len, seed, exact, ce, retrieval_top1, null_selected_rate, false_write_rate`

Why it matters:
This proves the model fails for the right reasons.

---

# Figure 8 — Training Dynamics

Filename:
`docs/figures/fig8_training_dynamics.png`

Panels:
A. answer exact vs step  
B. answer CE vs step  
C. writer recall vs step  
D. retrieval top1 vs step

Needed per-step columns:
`run_id, step, model, seq_len, seed, train_loss, eval_exact, eval_ce, writer_recall, retrieval_top1`

Why it matters:
This shows sample efficiency and learning behavior, not just final result.

---

# Priority order

1. Logging upgrade
2. Figure 1 multipanel
3. Figure 2 ablation heatmap
4. Figure 3 learned-writer diagnostics
5. Figure 8 training dynamics
6. Figure 4 retrieval score separation
7. Figure 7 failure controls
8. Figure 5 router/path usage
9. Figure 6 memory interaction graph

Do not start with the memory graph. It looks cool, but it requires trace logging first.
