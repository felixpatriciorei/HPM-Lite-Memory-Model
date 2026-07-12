"""Sharper version of check_template_variation.py.

That test proved V4 isn't locked to one fixed template. It didn't yet prove
V4 isn't just doing per-template classification-then-lookup across all 13
templates it was trained on -- a model doing that would also score 1.000
there.

This trains V4 on 8 templates only ("heavy": templates 1,3,4,7,8,9,10,11),
then evaluates on the 5 templates it has NEVER seen during training at all
(12,13,14,15,16 -- forced via a small override, since the config only
exposes nested template sets, not arbitrary held-out splits). Also evaluates
on the seen-template distribution as a same-run reference point.

Run the same way as before:
    python check_held_out_template.py
"""

from __future__ import annotations

import torch

from hpm_lite.data import FactRecallConfig, FactRecallDataset
from hpm_lite.noisy_extraction import ContextualTupleEdgeScorer, writer_quality_metrics

SEQ_LEN = 512
NUM_FACTS = 8
NUM_HARD_NEGATIVES = 2
CANDIDATE_K = 10
BATCH_SIZE = 32
TRAIN_STEPS = 1200
LR = 3e-4
TRAIN_SEED = 0
EVAL_SEED = 12345
EVAL_BATCHES = 8
HAS_CONDITION = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEN_TEMPLATES = [1, 3, 4, 7, 8, 9, 10, 11]       # what template_augmentation="heavy" trains on
HELD_OUT_TEMPLATES = [12, 13, 14, 15, 16]          # what "extreme" adds beyond "heavy" -- never trained on

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


class FixedTemplateDataset(FactRecallDataset):
    """Same generator, but _conditional_templates() is forced to a specific
    list regardless of the noise_level/template_augmentation config -- used
    to build a clean, 100% held-out-template eval set."""

    def __init__(self, config: FactRecallConfig, template_ids):
        super().__init__(config)
        self._forced_templates = list(template_ids)

    def _conditional_templates(self):
        return self._forced_templates


def make_train_dataset(seed: int) -> FactRecallDataset:
    cfg = FactRecallConfig(
        seq_len=SEQ_LEN,
        task="noisy_conditional",
        num_facts=NUM_FACTS,
        num_hard_negatives=NUM_HARD_NEGATIVES,
        seed=seed,
        marker_rate=0.0,
        noise_level="hard",
        template_augmentation="heavy",
    )
    return FactRecallDataset(cfg)


def make_eval_dataset(seed: int, template_ids) -> FactRecallDataset:
    cfg = FactRecallConfig(
        seq_len=SEQ_LEN,
        task="noisy_conditional",
        num_facts=NUM_FACTS,
        num_hard_negatives=NUM_HARD_NEGATIVES,
        seed=seed,
        marker_rate=0.0,
        noise_level="hard",
        template_augmentation="heavy",  # irrelevant, overridden below
    )
    return FixedTemplateDataset(cfg, template_ids)


def evaluate(model, ds: FactRecallDataset):
    model.eval()
    accum = {}
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            batch = ds.sample_batch(BATCH_SIZE, device=DEVICE)
            written = model.predict_batch(batch, CANDIDATE_K, "learned_candidates", decode_mode="threshold")
            metrics = writer_quality_metrics(batch, written, HAS_CONDITION)
            for k, v in metrics.items():
                accum.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in accum.items()}


REPORT_KEYS = ["key_accuracy", "condition_accuracy", "value_accuracy", "full_slot_exact", "all_slots_exact", "tuple_scoring_error_rate"]


def show(label: str, metrics: dict) -> None:
    parts = [f"{k}={metrics.get(k, float('nan')):.3f}" for k in REPORT_KEYS]
    print(f"  {label:22s} " + "  ".join(parts))


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Training templates (seen):   {SEEN_TEMPLATES}")
    print(f"Held-out templates (unseen): {HELD_OUT_TEMPLATES}\n")

    print("=" * 90)
    print("Training V4 on the 8 seen templates only")
    print("=" * 90)
    ds = make_train_dataset(TRAIN_SEED)
    model = ContextualTupleEdgeScorer(max_slots=NUM_FACTS, seq_len=SEQ_LEN, has_condition=HAS_CONDITION).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(TRAIN_STEPS):
        batch = ds.sample_batch(BATCH_SIZE, device=DEVICE)
        opt.zero_grad()
        loss = model.loss(batch, CANDIDATE_K)
        loss.backward()
        opt.step()
        if step % 300 == 0 or step == TRAIN_STEPS - 1:
            print(f"    step {step:5d}  loss {float(loss.item()):.4f}")

    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    seen_eval_ds = make_eval_dataset(EVAL_SEED, SEEN_TEMPLATES)
    held_out_eval_ds = make_eval_dataset(EVAL_SEED, HELD_OUT_TEMPLATES)
    show("seen templates", evaluate(model, seen_eval_ds))
    show("held-out templates", evaluate(model, held_out_eval_ds))

    print()
    print("What to look for:")
    print("  - Held-out close to seen: real content-based binding, transfers to new phrasing.")
    print("  - Held-out clearly worse but well above V3's ~0.35 ceiling: partial transfer --")
    print("    some real binding signal, but some template-specific pattern-matching too.")
    print("  - Held-out collapses near V3-level: this was template classification + lookup,")
    print("    not binding by content. Important to know before claiming generalization.")


if __name__ == "__main__":
    main()
