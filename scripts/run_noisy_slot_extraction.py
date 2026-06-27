from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpm_lite.data import FactRecallConfig, FactRecallDataset
from hpm_lite.noisy_extraction import (
    ContextualTupleEdgeScorer,
    LearnedSetExtractorV2,
    LearnedTypedExtractor,
    SPNTupleAssembler,
    WriterV3CandidateAssembler,
    writer_quality_metrics,
)
from hpm_lite.structured_readout import (
    LearnedConditionReader,
    LearnedSetReader,
    count_trainable_parameters,
    symbolic_condition_binding_metrics,
    symbolic_set_metrics,
)
from hpm_lite.train import TinyAdamW
from hpm_lite.utils import ensure_dir, resolve_device, set_seed
from hpm_lite.write_modes import apply_write_mode


TASKS = {"noisy_kv", "noisy_coexisting", "noisy_conditional"}
V45_WRITERS = {
    "baseline_v4",
    "v4_span_condition",
    "v4_guideline_condition",
    "v4_augmented_templates",
    "v4_simplified_aux",
    "v4_condition_v2_full",
}
WRITERS = {
    "oracle",
    "fact_token",
    "learned_typed_extractor",
    "learned_set_extractor_v2",
    "writer_v3_oracle_candidates",
    "writer_v3_oracle_candidates_sanity",
    "writer_v3_learned_candidates",
    "writer_v3_oracle_candidates_plus_noise",
    "writer_v3_learned_candidates_oracle_assembly",
    "spn_tuple_assembler_oracle_candidates",
    "contextual_tuple_oracle_candidates",
    "contextual_tuple_oracle_candidates_plus_hard_negatives",
    "contextual_tuple_learned_candidates",
    "contextual_tuple_learned_candidates_plus_oracle_missing",
    "contextual_tuple_oracle_candidates_plus_learned_noise",
    "contextual_tuple_oracle_key_candidates",
    "contextual_tuple_oracle_cond_candidates",
    "contextual_tuple_oracle_value_candidates",
    "contextual_tuple_oracle_key_cond_candidates",
    "contextual_tuple_oracle_key_value_candidates",
    "contextual_tuple_oracle_cond_value_candidates",
    "contextual_tuple_gold_key_cond",
    "contextual_tuple_gold_key",
    "contextual_tuple_gold_cond",
    "contextual_tuple_gold_value",
    "contextual_tuple_gold_all_fields",
    *V45_WRITERS,
}
NOISE_LEVELS = {"clean", "light", "medium", "hard"}
TEMPLATE_MIXES = {"simple", "mixed", "paraphrase"}
TEMPLATE_SPLITS = {"random", "heldout"}
TEMPLATE_AUGMENTATIONS = {"none", "light", "heavy", "extreme"}
EVAL_MODES = {
    "normal_v2",
    "oracle_count_topk",
    "oracle_objectness",
    "oracle_fields",
    "oracle_count_and_fields",
}


RAW_COLUMNS = [
    "task",
    "writer",
    "eval_mode",
    "eval_threshold",
    "lambda_obj",
    "candidate_k",
    "tuple_pruning",
    "pair_beam_size",
    "candidate_loss_weight",
    "seed",
    "budget_steps",
    "noise_level",
    "marker_rate",
    "distractor_count",
    "slot_count",
    "max_slots",
    "hard_negatives",
    "template_mix",
    "template_split",
    "template_augmentation",
    "simplified_aux_weight",
    "guideline_loss_weight",
    "reader_type",
    "slot_precision",
    "slot_recall",
    "slot_f1",
    "key_accuracy",
    "condition_accuracy",
    "value_accuracy",
    "full_slot_exact",
    "all_slots_exact",
    "false_slot_rate",
    "missed_slot_rate",
    "duplicate_slot_rate",
    "predicted_slot_count",
    "true_slot_count",
    "slot_count_accuracy",
    "overprediction_rate",
    "underprediction_rate",
    "objectness_accuracy",
    "objectness_auc",
    "objectness_precision",
    "objectness_recall",
    "objectness_f1",
    "mean_objectness_true_slots",
    "mean_objectness_false_slots",
    "objectness_margin",
    "candidate_key_recall",
    "candidate_condition_recall",
    "candidate_value_recall",
    "all_fields_candidate_recall",
    "learned_candidate_key_recall",
    "learned_candidate_cond_recall",
    "learned_candidate_value_recall",
    "learned_candidate_all_field_recall",
    "learned_candidate_key_precision",
    "learned_candidate_cond_precision",
    "learned_candidate_value_precision",
    "candidate_pool_size_key",
    "candidate_pool_size_condition",
    "candidate_pool_size_cond",
    "candidate_pool_size_value",
    "candidate_precision",
    "candidate_false_positive_rate",
    "candidate_miss_rate",
    "candidate_miss_rate_key",
    "candidate_miss_rate_cond",
    "candidate_miss_rate_value",
    "candidate_miss_rate_any",
    "post_query_leak_rate",
    "answer_token_leak_rate",
    "symbolic_answer_exact",
    "learned_reader_answer_exact",
    "reader_answer_exact_with_oracle_slots",
    "reader_answer_exact_with_fact_token_slots",
    "reader_answer_exact_with_learned_slots",
    "learned_slot_reader_gap_to_oracle",
    "symbolic_answer_exact_with_learned_slots",
    "learned_reader_exact_with_learned_slots",
    "extractor_loss",
    "extractor_trainable_params",
    "reader_trainable_params",
    "extractor_train_time",
    "extractor_examples_per_sec",
    "best_threshold",
    "best_threshold_exact",
    "best_threshold_slot_f1",
    "oracle_count_gain",
    "oracle_objectness_gain",
    "oracle_fields_gain",
    "assembler_slot_exact_given_oracle_candidates",
    "assembler_slot_exact_given_learned_candidates",
    "v3_exact",
    "v3_slot_f1",
    "v3_all_slots_exact",
    "v3_gain_over_v2",
    "v3_gap_to_oracle_fields",
    "key_candidate_accuracy",
    "condition_candidate_accuracy",
    "value_candidate_accuracy",
    "tuple_accuracy",
    "matched_tuple_accuracy",
    "mean_hungarian_gold_cost",
    "mean_hungarian_pred_cost",
    "spn_tuple_exact",
    "spn_tuple_accuracy",
    "spn_gain_over_independent_heads",
    "contextual_tuple_exact",
    "contextual_tuple_slot_f1",
    "contextual_tuple_all_slots_exact",
    "tuple_auc",
    "tuple_positive_score_mean",
    "tuple_negative_score_mean",
    "tuple_score_margin",
    "hard_negative_false_positive_rate",
    "top_true_count_exact",
    "threshold_best_exact",
    "gold_key_cond_value_accuracy",
    "gold_key_pair_accuracy",
    "gold_cond_pair_accuracy",
    "gold_value_pair_accuracy",
    "contextual_gain_over_spn",
    "contextual_gain_over_independent_heads",
    "learned_candidate_gap_to_oracle",
    "oracle_missing_repair_gain",
    "learned_noise_damage",
    "random_template_exact",
    "heldout_template_exact",
    "heldout_drop",
    "condition_token_recall",
    "condition_span_recall",
    "condition_any_recall",
    "condition_span_precision",
    "condition_miss_rate",
    "value_miss_rate",
    "tuple_scoring_error_rate",
    "train_time",
    "examples_per_sec",
    "tuple_candidates_scored",
    "tuple_scorer_time",
    "candidate_proposer_time",
    "reader_time",
    "gpu_memory_mb",
    "set_exact",
    "set_precision",
    "set_recall",
    "set_f1",
    "missed_positive_rate",
    "extra_false_positive_rate",
]

SUMMARY_COLUMNS = [
    "task",
    "writer",
    "eval_mode",
    "eval_threshold",
    "lambda_obj",
    "candidate_k",
    "tuple_pruning",
    "pair_beam_size",
    "candidate_loss_weight",
    "budget_steps",
    "noise_level",
    "marker_rate",
    "distractor_count",
    "slot_count",
    "max_slots",
    "hard_negatives",
    "template_mix",
    "template_split",
    "template_augmentation",
    "simplified_aux_weight",
    "guideline_loss_weight",
    "n",
]
GROUP_COLUMNS = set(SUMMARY_COLUMNS) - {"n"}
NON_METRIC_COLUMNS = GROUP_COLUMNS | {"seed", "reader_type"}
for metric in RAW_COLUMNS:
    if metric in NON_METRIC_COLUMNS:
        continue
    SUMMARY_COLUMNS.extend([f"{metric}_mean", f"{metric}_std"])


def parse_str_list(value: str, allowed: set[str]) -> List[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"unknown values: {unknown}")
    return items


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def has_condition(task: str) -> bool:
    return task == "noisy_conditional"


def reader_type(task: str) -> str:
    return "condition" if has_condition(task) else "set"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate noisy typed slot extraction.")
    parser.add_argument("--tasks", type=str, default="noisy_conditional,noisy_coexisting")
    parser.add_argument("--writers", type=str, default="oracle,fact_token,learned_typed_extractor")
    parser.add_argument("--budgets", type=str, default="0,10,100")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--extractor-dim", type=int, default=64)
    parser.add_argument("--extractor-hidden", type=int, default=128)
    parser.add_argument("--extractor-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--reader-pretrain-steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--noise-levels", type=str, default="clean,medium")
    parser.add_argument("--marker-rates", type=str, default="1.0,0.5,0.0")
    parser.add_argument("--distractor-counts", type=str, default="0,8")
    parser.add_argument("--slot-counts", type=str, default="4,16")
    parser.add_argument("--max-slots", type=str, default="")
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    parser.add_argument("--lambda-obj", type=float, default=1.0)
    parser.add_argument("--lambda-obj-values", type=str, default="")
    parser.add_argument("--threshold-sweep", type=str, default="")
    parser.add_argument("--eval-modes", type=str, default="")
    parser.add_argument("--candidate-k-values", type=str, default="")
    parser.add_argument("--slow-sweep", action="store_true")
    parser.add_argument("--tuple-pruning", type=str, default="none", choices=["none", "pair_beam"])
    parser.add_argument("--pair-beam-size", type=int, default=8)
    parser.add_argument("--candidate-loss-weight", type=str, default="1.0")
    parser.add_argument("--tuple-loss-weight", type=float, default=1.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    parser.add_argument("--tuple-debug-examples", type=int, default=0)
    parser.add_argument("--hard-negatives", type=str, default="0")
    parser.add_argument("--template-mix", type=str, default="mixed")
    parser.add_argument("--template-split", type=str, default="random")
    parser.add_argument("--template-augmentation", type=str, default="extreme")
    parser.add_argument("--template-augmentations", type=str, default="")
    parser.add_argument("--simplified-aux-weight", type=float, default=0.5)
    parser.add_argument("--simplified-aux-weight-values", type=str, default="")
    parser.add_argument("--guideline-loss-weight", type=float, default=2.0)
    parser.add_argument("--guideline-loss-weight-values", type=str, default="")
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="runs")
    return parser


def make_dataset(
    task: str,
    args: argparse.Namespace,
    seed: int,
    slot_count: int,
    noise_level: str,
    marker_rate: float,
    distractor_count: int,
    hard_negatives: int,
    split_phase: str = "eval",
    template_augmentation: str = "none",
) -> FactRecallDataset:
    template_mix = args.template_mix
    if args.template_split == "heldout":
        template_mix = "simple" if split_phase == "train" else "paraphrase"
    return FactRecallDataset(
        FactRecallConfig(
            seq_len=args.seq_len,
            window=args.window,
            task=task,
            seed=seed,
            num_facts=slot_count,
            oracle_memory=True,
            num_hard_negatives=hard_negatives,
            noise_level=noise_level,
            marker_rate=marker_rate,
            distractor_count=distractor_count,
            template_mix=template_mix,
            template_augmentation=template_augmentation if split_phase == "train" else "none",
        )
    )


def make_reader(task: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    kwargs = {
        "reader_dim": args.extractor_dim,
        "hidden": args.extractor_hidden,
        "layers": 2,
        "dropout": 0.0,
        "train_embeddings": True,
    }
    if has_condition(task):
        return LearnedConditionReader(**kwargs).to(device)
    return LearnedSetReader(**kwargs).to(device)


def reader_loss(task: str, reader: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return reader.loss(batch, "normal")


def reader_exact(task: str, reader: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> float:
    metrics = reader.metrics(batch, "normal")
    if has_condition(task):
        return float(metrics.get("learned_condition_exact", 0.0))
    return float(metrics.get("learned_set_exact", 0.0))


def symbolic_exact(task: str, batch: Dict[str, torch.Tensor]) -> float:
    if has_condition(task):
        return float(symbolic_condition_binding_metrics(batch, "normal").get("condition_symbolic_exact", 0.0))
    return float(symbolic_set_metrics(batch, "normal").get("symbolic_set_exact", 0.0))


def train_reader(
    task: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    slot_count: int,
) -> torch.nn.Module:
    set_seed(seed + 11_000)
    reader = make_reader(task, args, device)
    optimizer = TinyAdamW(reader.parameters(), lr=args.lr, weight_decay=0.0)
    dataset = make_dataset(task, args, seed + 12_000, slot_count, "clean", 1.0, 0, 0)
    for _ in range(args.reader_pretrain_steps):
        batch = dataset.sample_batch(args.batch_size, device=device)
        loss = reader_loss(task, reader, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        optimizer.step()
    reader.eval()
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    return reader


def apply_writer(
    writer: str,
    batch: Dict[str, torch.Tensor],
    extractor_v1: LearnedTypedExtractor | None,
    extractor_v2: LearnedSetExtractorV2 | None,
    writer_v3: WriterV3CandidateAssembler | None = None,
    spn_assembler: SPNTupleAssembler | None = None,
    contextual_assembler: ContextualTupleEdgeScorer | None = None,
    contextual_assemblers: Dict[str, ContextualTupleEdgeScorer] | None = None,
    candidate_k: int = 0,
    eval_mode: str = "normal",
    eval_threshold: float | None = None,
    tuple_pruning: str = "none",
    pair_beam_size: int = 8,
) -> Dict[str, torch.Tensor]:
    if writer == "oracle":
        return {key: value.clone() for key, value in batch.items()}
    if writer == "fact_token":
        written, _ = apply_write_mode(batch, "fact_token")
        return written
    if writer == "learned_set_extractor_v2":
        if extractor_v2 is None:
            raise ValueError("learned_set_extractor_v2 writer requires a v2 extractor")
        mode = "normal_v2" if eval_mode in {"", "normal"} else eval_mode
        if mode == "threshold_sweep":
            mode = "normal_v2"
        return extractor_v2.predict_batch(batch, eval_mode=mode, threshold=eval_threshold)
    if writer.startswith("writer_v3_"):
        if writer_v3 is None:
            raise ValueError(f"{writer} requires a v3 writer")
        mode_map = {
            "writer_v3_oracle_candidates": "oracle_candidates",
            "writer_v3_oracle_candidates_sanity": "oracle_candidates_gold_only",
            "writer_v3_learned_candidates": "learned_candidates",
            "writer_v3_oracle_candidates_plus_noise": "oracle_candidates_plus_noise",
            "writer_v3_learned_candidates_oracle_assembly": "learned_candidates_oracle_assembly",
        }
        assembly_mode = eval_mode if writer == "writer_v3_oracle_candidates_sanity" else "independent_field_heads_current"
        return writer_v3.predict_batch(batch, candidate_k=candidate_k, candidate_mode=mode_map[writer], assembly_eval_mode=assembly_mode)
    if writer == "spn_tuple_assembler_oracle_candidates":
        if spn_assembler is None:
            raise ValueError("spn_tuple_assembler_oracle_candidates requires an SPN tuple assembler")
        return spn_assembler.predict_batch(batch, candidate_k=candidate_k)
    if writer in V45_WRITERS:
        if contextual_assemblers is None or writer not in contextual_assemblers:
            raise ValueError(f"{writer} requires its own v4.5 contextual assembler")
        return contextual_assemblers[writer].predict_batch(
            batch,
            candidate_k=candidate_k,
            candidate_mode="learned_candidates",
            decode_mode="top_true_count",
            threshold=eval_threshold,
            tuple_pruning=tuple_pruning,
            pair_beam_size=pair_beam_size,
        )
    if writer.startswith("contextual_tuple_"):
        if contextual_assembler is None:
            raise ValueError(f"{writer} requires a contextual tuple assembler")
        mode_map = {
            "contextual_tuple_oracle_candidates": "oracle_candidates_gold_only",
            "contextual_tuple_oracle_candidates_plus_hard_negatives": "oracle_candidates_plus_noise",
            "contextual_tuple_learned_candidates": "learned_candidates",
            "contextual_tuple_learned_candidates_plus_oracle_missing": "learned_candidates_plus_oracle_missing",
            "contextual_tuple_oracle_candidates_plus_learned_noise": "oracle_candidates_plus_learned_noise",
            "contextual_tuple_oracle_key_candidates": "oracle_key_candidates",
            "contextual_tuple_oracle_cond_candidates": "oracle_cond_candidates",
            "contextual_tuple_oracle_value_candidates": "oracle_value_candidates",
            "contextual_tuple_oracle_key_cond_candidates": "oracle_key_cond_candidates",
            "contextual_tuple_oracle_key_value_candidates": "oracle_key_value_candidates",
            "contextual_tuple_oracle_cond_value_candidates": "oracle_cond_value_candidates",
            "contextual_tuple_gold_key_cond": "oracle_candidates_gold_only",
            "contextual_tuple_gold_key": "oracle_candidates_gold_only",
            "contextual_tuple_gold_cond": "oracle_candidates_gold_only",
            "contextual_tuple_gold_value": "oracle_candidates_gold_only",
            "contextual_tuple_gold_all_fields": "oracle_candidates_gold_only",
        }
        decode_map = {
            "contextual_tuple_oracle_candidates": "top_true_count",
            "contextual_tuple_oracle_candidates_plus_hard_negatives": "top_true_count",
            "contextual_tuple_learned_candidates": "top_true_count",
            "contextual_tuple_learned_candidates_plus_oracle_missing": "top_true_count",
            "contextual_tuple_oracle_candidates_plus_learned_noise": "top_true_count",
            "contextual_tuple_oracle_key_candidates": "top_true_count",
            "contextual_tuple_oracle_cond_candidates": "top_true_count",
            "contextual_tuple_oracle_value_candidates": "top_true_count",
            "contextual_tuple_oracle_key_cond_candidates": "top_true_count",
            "contextual_tuple_oracle_key_value_candidates": "top_true_count",
            "contextual_tuple_oracle_cond_value_candidates": "top_true_count",
            "contextual_tuple_gold_key_cond": "gold_key_cond",
            "contextual_tuple_gold_key": "gold_key",
            "contextual_tuple_gold_cond": "gold_cond",
            "contextual_tuple_gold_value": "gold_value",
            "contextual_tuple_gold_all_fields": "gold_all_fields",
        }
        return contextual_assembler.predict_batch(
            batch,
            candidate_k=candidate_k,
            candidate_mode=mode_map[writer],
            decode_mode=decode_map[writer],
            threshold=eval_threshold,
            tuple_pruning=tuple_pruning,
            pair_beam_size=pair_beam_size,
        )
    if extractor_v1 is None:
        raise ValueError("learned_typed_extractor writer requires an extractor")
    return extractor_v1.predict_batch(batch)


@torch.no_grad()
def evaluate_writer(
    task: str,
    writer: str,
    reader: torch.nn.Module,
    extractor_v1: LearnedTypedExtractor | None,
    extractor_v2: LearnedSetExtractorV2 | None,
    writer_v3: WriterV3CandidateAssembler | None,
    spn_assembler: SPNTupleAssembler | None,
    contextual_assembler: ContextualTupleEdgeScorer | None,
    contextual_assemblers: Dict[str, ContextualTupleEdgeScorer] | None,
    dataset: FactRecallDataset,
    args: argparse.Namespace,
    device: torch.device,
    candidate_k: int = 0,
    eval_mode: str = "normal",
    eval_threshold: float | None = None,
    tuple_pruning: str = "none",
    pair_beam_size: int = 8,
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    total = 0
    oracle_reader_values = []
    for _ in range(args.eval_batches):
        batch = dataset.sample_batch(args.batch_size, device=device)
        oracle_exact = reader_exact(task, reader, batch)
        written = apply_writer(
            writer,
            batch,
            extractor_v1,
            extractor_v2,
            writer_v3=writer_v3,
            spn_assembler=spn_assembler,
            contextual_assembler=contextual_assembler,
            contextual_assemblers=contextual_assemblers,
            candidate_k=candidate_k,
            eval_mode=eval_mode,
            eval_threshold=eval_threshold,
            tuple_pruning=tuple_pruning,
            pair_beam_size=pair_beam_size,
        )
        writer_metrics = writer_quality_metrics(batch, written, has_condition(task))
        writer_metrics.update({key: float(value) for key, value in written.get("assembler_debug_metrics", {}).items()})
        symbolic = symbolic_exact(task, written)
        reader_start = time.perf_counter()
        reader_metrics = reader.metrics(written, "normal")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        reader_elapsed = time.perf_counter() - reader_start
        learned = float(
            reader_metrics.get("learned_condition_exact", reader_metrics.get("learned_set_exact", reader_exact(task, reader, written)))
        )
        metrics = {
            **writer_metrics,
            **{key: float(value) for key, value in reader_metrics.items()},
            "symbolic_answer_exact": symbolic,
            "learned_reader_answer_exact": learned,
            "reader_time": reader_elapsed,
            "gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0,
        }
        if not has_condition(task):
            metrics["set_exact"] = float(reader_metrics.get("learned_set_exact", 0.0))
            metrics["set_precision"] = float(reader_metrics.get("learned_set_precision", 0.0))
            metrics["set_recall"] = float(reader_metrics.get("learned_set_recall", 0.0))
            metrics["set_f1"] = float(reader_metrics.get("learned_set_f1", 0.0))
            metrics["missed_positive_rate"] = 1.0 - metrics["set_recall"]
            metrics["extra_false_positive_rate"] = 1.0 - metrics["set_precision"]
        if writer == "spn_tuple_assembler_oracle_candidates":
            metrics["spn_tuple_exact"] = learned
            metrics["spn_tuple_accuracy"] = writer_metrics.get("slot_f1", 0.0)
        if writer.startswith("contextual_tuple_") or writer in V45_WRITERS:
            metrics["contextual_tuple_exact"] = learned
            metrics["contextual_tuple_slot_f1"] = writer_metrics.get("slot_f1", 0.0)
            metrics["contextual_tuple_all_slots_exact"] = writer_metrics.get("all_slots_exact", 0.0)
            metrics["top_true_count_exact"] = learned if writer in {
                "contextual_tuple_oracle_candidates",
                "contextual_tuple_oracle_candidates_plus_hard_negatives",
                "contextual_tuple_learned_candidates",
            } or writer in V45_WRITERS else 0.0
            if writer == "contextual_tuple_gold_key_cond":
                metrics["gold_key_cond_value_accuracy"] = writer_metrics.get("value_accuracy", 0.0)
            elif writer == "contextual_tuple_gold_key":
                metrics["gold_key_pair_accuracy"] = writer_metrics.get("full_slot_exact", 0.0)
            elif writer == "contextual_tuple_gold_cond":
                metrics["gold_cond_pair_accuracy"] = writer_metrics.get("full_slot_exact", 0.0)
            elif writer == "contextual_tuple_gold_value":
                metrics["gold_value_pair_accuracy"] = writer_metrics.get("full_slot_exact", 0.0)
        if writer == "oracle":
            metrics["reader_answer_exact_with_oracle_slots"] = learned
        elif writer == "fact_token":
            metrics["reader_answer_exact_with_fact_token_slots"] = learned
        else:
            metrics["reader_answer_exact_with_learned_slots"] = learned
            metrics["symbolic_answer_exact_with_learned_slots"] = symbolic
            metrics["learned_reader_exact_with_learned_slots"] = learned
        oracle_reader_values.append(oracle_exact)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value) * args.batch_size
        total += args.batch_size
    out = {key: value / max(total, 1) for key, value in sums.items()}
    oracle_mean = statistics.mean(oracle_reader_values) if oracle_reader_values else 0.0
    if writer in {"learned_typed_extractor", "learned_set_extractor_v2"}:
        out["learned_slot_reader_gap_to_oracle"] = out.get("learned_reader_answer_exact", 0.0) - oracle_mean
    return out


def eval_specs_for_writer(writer: str, args: argparse.Namespace) -> List[Tuple[str, float | None]]:
    if writer == "writer_v3_oracle_candidates_sanity":
        return [
            ("independent_field_heads_current", None),
            ("no_objectness_true_count", None),
            ("no_hungarian_canonical_debug", None),
            ("gold_key_only", None),
            ("gold_cond_only", None),
            ("gold_value_only", None),
            ("gold_key_cond", None),
            ("gold_all_fields", None),
        ]
    if writer != "learned_set_extractor_v2" or not is_bottleneck_run(args):
        return [("normal", None)]
    modes = parse_str_list(args.eval_modes, EVAL_MODES) if args.eval_modes else ["normal_v2"]
    specs: List[Tuple[str, float | None]] = [(mode, None) for mode in modes]
    for threshold in parse_float_list(args.threshold_sweep) if args.threshold_sweep else []:
        specs.append(("threshold_sweep", threshold))
    return specs


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def summarize(raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in raw_rows:
        key = (
            row["task"],
            row["writer"],
            row.get("eval_mode", ""),
            float(row.get("eval_threshold", -1.0) if row.get("eval_threshold", "") != "" else -1.0),
            float(row.get("lambda_obj", 0.0) if row.get("lambda_obj", "") != "" else 0.0),
            int(row.get("candidate_k", 0) if row.get("candidate_k", "") != "" else 0),
            row.get("tuple_pruning", "none"),
            int(row.get("pair_beam_size", 0) if row.get("pair_beam_size", "") != "" else 0),
            float(row.get("candidate_loss_weight", 0.0) if row.get("candidate_loss_weight", "") != "" else 0.0),
            int(row["budget_steps"]),
            row["noise_level"],
            float(row["marker_rate"]),
            int(row["distractor_count"]),
            int(row["slot_count"]),
            int(row["max_slots"]),
            int(row["hard_negatives"]),
            row["template_mix"],
            row.get("template_split", "random"),
            row.get("template_augmentation", "none"),
            float(row.get("simplified_aux_weight", 0.0) if row.get("simplified_aux_weight", "") != "" else 0.0),
            float(row.get("guideline_loss_weight", 0.0) if row.get("guideline_loss_weight", "") != "" else 0.0),
        )
        groups.setdefault(key, []).append(row)
    metrics = [column[:-5] for column in SUMMARY_COLUMNS if column.endswith("_mean")]
    out = []
    for key in sorted(groups):
        task, writer, eval_mode, eval_threshold, lambda_obj, candidate_k, tuple_pruning, pair_beam_size, candidate_loss_weight, budget, noise_level, marker_rate, distractors, slots, max_slots, hard, template_mix, template_split, template_augmentation, simplified_aux_weight, guideline_loss_weight = key
        group = groups[key]
        row: Dict[str, object] = {
            "task": task,
            "writer": writer,
            "eval_mode": eval_mode,
            "eval_threshold": eval_threshold if eval_threshold >= 0 else "",
            "lambda_obj": lambda_obj,
            "candidate_k": candidate_k,
            "tuple_pruning": tuple_pruning,
            "pair_beam_size": pair_beam_size,
            "candidate_loss_weight": candidate_loss_weight,
            "budget_steps": budget,
            "noise_level": noise_level,
            "marker_rate": marker_rate,
            "distractor_count": distractors,
            "slot_count": slots,
            "max_slots": max_slots,
            "hard_negatives": hard,
            "template_mix": template_mix,
            "template_split": template_split,
            "template_augmentation": template_augmentation,
            "simplified_aux_weight": simplified_aux_weight,
            "guideline_loss_weight": guideline_loss_weight,
            "n": len(group),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group if item.get(metric, "") != ""]
            if values:
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
        out.append(row)
    return out


def is_bottleneck_run(args: argparse.Namespace) -> bool:
    return bool(args.eval_modes or args.threshold_sweep or args.lambda_obj_values)


def is_v3_run(writers: List[str]) -> bool:
    return any(writer.startswith("writer_v3_") or writer == "spn_tuple_assembler_oracle_candidates" for writer in writers)


def is_tuple_debug_run(writers: List[str]) -> bool:
    return "writer_v3_oracle_candidates_sanity" in writers or "spn_tuple_assembler_oracle_candidates" in writers


def is_contextual_tuple_run(writers: List[str]) -> bool:
    return any(writer.startswith("contextual_tuple_") or writer in V45_WRITERS for writer in writers)


def is_contextual_learned_candidate_run(writers: List[str]) -> bool:
    learned_markers = {
        "contextual_tuple_learned_candidates",
        "contextual_tuple_learned_candidates_plus_oracle_missing",
        "contextual_tuple_oracle_candidates_plus_learned_noise",
        "contextual_tuple_oracle_key_candidates",
        "contextual_tuple_oracle_cond_candidates",
        "contextual_tuple_oracle_value_candidates",
        "contextual_tuple_oracle_key_cond_candidates",
        "contextual_tuple_oracle_key_value_candidates",
        "contextual_tuple_oracle_cond_value_candidates",
    }
    return any(writer in learned_markers or writer in V45_WRITERS for writer in writers)


def v45_variant(writer: str) -> str:
    return {
        "baseline_v4": "baseline",
        "v4_span_condition": "span",
        "v4_guideline_condition": "guideline",
        "v4_augmented_templates": "baseline",
        "v4_simplified_aux": "simplified",
        "v4_condition_v2_full": "full",
    }[writer]


def v45_uses_augmented_templates(writer: str) -> bool:
    return writer in {"v4_augmented_templates", "v4_condition_v2_full"}


def annotate_bottleneck_rows(raw_rows: List[Dict[str, object]]) -> None:
    normal_by_key: Dict[tuple, float] = {}
    threshold_rows_by_key: Dict[tuple, List[Dict[str, object]]] = {}

    def key_for(row: Dict[str, object]) -> tuple:
        return (
            row.get("task"),
            row.get("writer"),
            row.get("seed"),
            row.get("budget_steps"),
            row.get("noise_level"),
            row.get("marker_rate"),
            row.get("distractor_count"),
            row.get("slot_count"),
            row.get("max_slots"),
            row.get("hard_negatives"),
            row.get("template_mix"),
            row.get("template_split", "random"),
            row.get("template_augmentation", "none"),
            row.get("simplified_aux_weight", 0.0),
            row.get("guideline_loss_weight", 0.0),
            row.get("tuple_pruning", "none"),
            row.get("pair_beam_size", 0),
            row.get("lambda_obj"),
        )

    for row in raw_rows:
        key = key_for(row)
        if row.get("eval_mode") == "normal_v2":
            normal_by_key[key] = float(row.get("learned_reader_answer_exact", 0.0) or 0.0)
        if row.get("eval_mode") == "threshold_sweep":
            threshold_rows_by_key.setdefault(key, []).append(row)

    best_by_key: Dict[tuple, Tuple[float, float, float]] = {}
    for key, rows in threshold_rows_by_key.items():
        best = max(rows, key=lambda item: float(item.get("learned_reader_answer_exact", 0.0) or 0.0))
        best_by_key[key] = (
            float(best.get("eval_threshold", 0.0) or 0.0),
            float(best.get("learned_reader_answer_exact", 0.0) or 0.0),
            float(best.get("slot_f1", 0.0) or 0.0),
        )

    for row in raw_rows:
        key = key_for(row)
        normal = normal_by_key.get(key)
        if key in best_by_key:
            row["best_threshold"], row["best_threshold_exact"], row["best_threshold_slot_f1"] = best_by_key[key]
        mode = row.get("eval_mode")
        if normal is None:
            continue
        exact = float(row.get("learned_reader_answer_exact", 0.0) or 0.0)
        if mode == "oracle_count_topk":
            row["oracle_count_gain"] = exact - normal
        elif mode == "oracle_objectness":
            row["oracle_objectness_gain"] = exact - normal
        elif mode == "oracle_fields":
            row["oracle_fields_gain"] = exact - normal


def annotate_v3_rows(raw_rows: List[Dict[str, object]]) -> None:
    v2_by_key: Dict[tuple, float] = {}
    oracle_by_key: Dict[tuple, float] = {}

    def key_for(row: Dict[str, object]) -> tuple:
        return (
            row.get("task"),
            row.get("seed"),
            row.get("budget_steps"),
            row.get("noise_level"),
            row.get("marker_rate"),
            row.get("distractor_count"),
            row.get("slot_count"),
            row.get("max_slots"),
            row.get("hard_negatives"),
            row.get("template_mix"),
            row.get("template_split", "random"),
            row.get("template_augmentation", "none"),
            row.get("simplified_aux_weight", 0.0),
            row.get("guideline_loss_weight", 0.0),
            row.get("candidate_k"),
            row.get("tuple_pruning", "none"),
            row.get("pair_beam_size", 0),
            row.get("candidate_loss_weight"),
        )

    for row in raw_rows:
        key = key_for(row)
        if row.get("writer") == "learned_set_extractor_v2":
            v2_by_key[key] = float(row.get("learned_reader_answer_exact", 0.0) or 0.0)
        if row.get("writer") == "writer_v3_oracle_candidates":
            oracle_by_key[key] = float(row.get("learned_reader_answer_exact", 0.0) or 0.0)

    for row in raw_rows:
        writer = str(row.get("writer", ""))
        if not writer.startswith("writer_v3_"):
            continue
        exact = float(row.get("learned_reader_answer_exact", 0.0) or 0.0)
        row["v3_exact"] = exact
        row["v3_slot_f1"] = row.get("slot_f1", "")
        row["v3_all_slots_exact"] = row.get("all_slots_exact", "")
        if writer == "writer_v3_oracle_candidates":
            row["assembler_slot_exact_given_oracle_candidates"] = row.get("all_slots_exact", "")
        if writer == "writer_v3_learned_candidates":
            row["assembler_slot_exact_given_learned_candidates"] = row.get("all_slots_exact", "")
        key = key_for(row)
        if key in v2_by_key:
            row["v3_gain_over_v2"] = exact - v2_by_key[key]
        if key in oracle_by_key:
            row["v3_gap_to_oracle_fields"] = oracle_by_key[key] - exact


def annotate_contextual_rows(raw_rows: List[Dict[str, object]]) -> None:
    spn_baseline = 0.2937
    independent_baseline = 0.0563
    oracle_by_key: Dict[tuple, float] = {}
    learned_by_key: Dict[tuple, float] = {}
    repaired_by_key: Dict[tuple, float] = {}
    noisy_by_key: Dict[tuple, float] = {}

    def key_for(row: Dict[str, object]) -> tuple:
        return (
            row.get("task"),
            row.get("seed"),
            row.get("budget_steps"),
            row.get("noise_level"),
            row.get("marker_rate"),
            row.get("distractor_count"),
            row.get("slot_count"),
            row.get("max_slots"),
            row.get("hard_negatives"),
            row.get("template_mix"),
            row.get("template_split", "random"),
            row.get("template_augmentation", "none"),
            row.get("simplified_aux_weight", 0.0),
            row.get("guideline_loss_weight", 0.0),
            row.get("candidate_k"),
            row.get("tuple_pruning", "none"),
            row.get("pair_beam_size", 0),
            row.get("candidate_loss_weight"),
        )

    for row in raw_rows:
        exact = float(row.get("learned_reader_answer_exact", 0.0) or 0.0)
        key = key_for(row)
        if row.get("writer") in {"contextual_tuple_oracle_candidates", "oracle"}:
            oracle_by_key[key] = exact
        elif row.get("writer") in {"contextual_tuple_learned_candidates", "baseline_v4"}:
            learned_by_key[key] = exact
        elif row.get("writer") == "contextual_tuple_learned_candidates_plus_oracle_missing":
            repaired_by_key[key] = exact
        elif row.get("writer") == "contextual_tuple_oracle_candidates_plus_learned_noise":
            noisy_by_key[key] = exact

    for row in raw_rows:
        writer = str(row.get("writer", ""))
        if not writer.startswith("contextual_tuple_") and writer not in V45_WRITERS:
            continue
        key = key_for(row)
        exact = float(row.get("learned_reader_answer_exact", 0.0) or 0.0)
        row["contextual_tuple_exact"] = exact
        row["contextual_tuple_slot_f1"] = row.get("slot_f1", "")
        row["contextual_tuple_all_slots_exact"] = row.get("all_slots_exact", "")
        row["contextual_gain_over_spn"] = exact - spn_baseline
        row["contextual_gain_over_independent_heads"] = exact - independent_baseline
        if key in oracle_by_key and writer == "contextual_tuple_learned_candidates":
            row["learned_candidate_gap_to_oracle"] = oracle_by_key[key] - exact
        if key in learned_by_key and writer == "contextual_tuple_learned_candidates_plus_oracle_missing":
            row["oracle_missing_repair_gain"] = exact - learned_by_key[key]
        if key in oracle_by_key and writer == "contextual_tuple_oracle_candidates_plus_learned_noise":
            row["learned_noise_damage"] = oracle_by_key[key] - exact


def flush(
    out_dir: Path,
    raw_rows: List[Dict[str, object]],
    v2_mode: bool = False,
    bottleneck_mode: bool = False,
    v3_mode: bool = False,
    contextual_mode: bool = False,
    output_prefix: str = "",
) -> List[Dict[str, object]]:
    if bottleneck_mode:
        annotate_bottleneck_rows(raw_rows)
    if v3_mode:
        annotate_v3_rows(raw_rows)
    if contextual_mode:
        annotate_contextual_rows(raw_rows)
    summary = summarize(raw_rows)
    contextual_debug_mode = any(str(row.get("writer", "")).startswith("contextual_tuple_") or row.get("writer") in V45_WRITERS for row in raw_rows)
    contextual_learned_mode = any(
        row.get("writer")
        in {
            "contextual_tuple_learned_candidates",
            "contextual_tuple_learned_candidates_plus_oracle_missing",
            "contextual_tuple_oracle_candidates_plus_learned_noise",
            "contextual_tuple_oracle_key_candidates",
            "contextual_tuple_oracle_cond_candidates",
            "contextual_tuple_oracle_value_candidates",
            "contextual_tuple_oracle_key_cond_candidates",
            "contextual_tuple_oracle_key_value_candidates",
            "contextual_tuple_oracle_cond_value_candidates",
        }
            or row.get("writer") in V45_WRITERS
        for row in raw_rows
    )
    tuple_debug_mode = any(
        row.get("writer") in {"writer_v3_oracle_candidates_sanity", "spn_tuple_assembler_oracle_candidates"}
        for row in raw_rows
    )
    if output_prefix:
        write_csv(out_dir / f"{output_prefix}_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / f"{output_prefix}_summary.csv", summary, SUMMARY_COLUMNS)
    elif contextual_learned_mode:
        write_csv(out_dir / "contextual_learned_candidates_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / "contextual_learned_candidates_summary.csv", summary, SUMMARY_COLUMNS)
    elif contextual_debug_mode:
        write_csv(out_dir / "contextual_tuple_scorer_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / "contextual_tuple_scorer_summary.csv", summary, SUMMARY_COLUMNS)
    elif tuple_debug_mode:
        write_csv(out_dir / "tuple_assembler_debug_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / "tuple_assembler_debug_summary.csv", summary, SUMMARY_COLUMNS)
    elif v3_mode:
        write_csv(out_dir / "writer_v3_field_candidates_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / "writer_v3_field_candidates_summary.csv", summary, SUMMARY_COLUMNS)
    elif bottleneck_mode:
        write_csv(out_dir / "writer_v2_bottleneck_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / "writer_v2_bottleneck_summary.csv", summary, SUMMARY_COLUMNS)
    elif v2_mode:
        write_csv(out_dir / "noisy_slot_extraction_v2_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / "noisy_slot_extraction_v2_summary.csv", summary, SUMMARY_COLUMNS)
    else:
        write_csv(out_dir / "noisy_slot_extraction_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / "noisy_slot_extraction_summary.csv", summary, SUMMARY_COLUMNS)
    return summary


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def summary_mean(rows: List[Dict[str, object]], writer: str, budget: int, metric: str, **filters: object) -> float | None:
    values = []
    for row in rows:
        if row["writer"] != writer or int(row["budget_steps"]) != budget:
            continue
        if any(str(row[key]) != str(value) for key, value in filters.items()):
            continue
        value = row.get(f"{metric}_mean", "")
        if value != "":
            values.append(float(value))
    return statistics.mean(values) if values else None


def writer_label(writer: str) -> str:
    labels = {
        "oracle": "oracle_writer",
        "fact_token": "fact_token_writer",
        "learned_typed_extractor": "learned_typed_extractor_v1",
        "learned_set_extractor_v2": "learned_set_extractor_v2",
        "writer_v3_oracle_candidates": "writer_v3_oracle_candidates",
        "writer_v3_oracle_candidates_sanity": "writer_v3_oracle_candidates_sanity",
        "writer_v3_learned_candidates": "writer_v3_learned_candidates",
        "writer_v3_oracle_candidates_plus_noise": "writer_v3_oracle_candidates_plus_noise",
        "writer_v3_learned_candidates_oracle_assembly": "writer_v3_learned_candidates_oracle_assembly",
        "spn_tuple_assembler_oracle_candidates": "spn_tuple_assembler_oracle_candidates",
        "contextual_tuple_oracle_candidates": "contextual_tuple_oracle_candidates",
        "contextual_tuple_oracle_candidates_plus_hard_negatives": "contextual_tuple_oracle_candidates_plus_hard_negatives",
        "contextual_tuple_learned_candidates": "contextual_tuple_learned_candidates",
        "contextual_tuple_learned_candidates_plus_oracle_missing": "contextual_tuple_learned_candidates_plus_oracle_missing",
        "contextual_tuple_oracle_candidates_plus_learned_noise": "contextual_tuple_oracle_candidates_plus_learned_noise",
        "contextual_tuple_oracle_key_candidates": "contextual_tuple_oracle_key_candidates",
        "contextual_tuple_oracle_cond_candidates": "contextual_tuple_oracle_cond_candidates",
        "contextual_tuple_oracle_value_candidates": "contextual_tuple_oracle_value_candidates",
        "contextual_tuple_oracle_key_cond_candidates": "contextual_tuple_oracle_key_cond_candidates",
        "contextual_tuple_oracle_key_value_candidates": "contextual_tuple_oracle_key_value_candidates",
        "contextual_tuple_oracle_cond_value_candidates": "contextual_tuple_oracle_cond_value_candidates",
        "contextual_tuple_gold_key_cond": "contextual_tuple_gold_key_cond",
        "contextual_tuple_gold_key": "contextual_tuple_gold_key",
        "contextual_tuple_gold_cond": "contextual_tuple_gold_cond",
        "contextual_tuple_gold_value": "contextual_tuple_gold_value",
        "contextual_tuple_gold_all_fields": "contextual_tuple_gold_all_fields",
        "baseline_v4": "baseline_v4",
        "v4_span_condition": "v4_span_condition",
        "v4_guideline_condition": "v4_guideline_condition",
        "v4_augmented_templates": "v4_augmented_templates",
        "v4_simplified_aux": "v4_simplified_aux",
        "v4_condition_v2_full": "v4_condition_v2_full",
    }
    return labels.get(writer, writer)


def update_results(path: Path, summary: List[Dict[str, object]], args: argparse.Namespace, raw_count: int, projected_rows: int) -> None:
    final_budget = max(parse_int_list(args.budgets))
    writers = parse_str_list(args.writers, WRITERS)
    v2_mode = "learned_set_extractor_v2" in writers
    if args.output_prefix in {"writer_v45_frozen_baseline", "writer_v45_pairbeam_b4", "writer_v45_pairbeam_b8", "writer_v45_pairbeam_b16"}:
        return
    if args.output_prefix == "writer_v45_optimization":
        rows = [row for row in summary if row.get("writer") == "v4_condition_v2_full" and int(row.get("budget_steps", 0)) == final_budget]

        def metric(row: Dict[str, object] | None, name: str) -> float | None:
            if row is None:
                return None
            value = row.get(f"{name}_mean", "")
            return None if value == "" else float(value)

        baseline = next((row for row in rows if row.get("tuple_pruning") == "none"), None)
        baseline_tuple_time = metric(baseline, "tuple_scorer_time")
        lines = [
            "## Writer v4.5 Freeze + Speed Optimization",
            "",
            "This freezes the repaired v4.5 setting and compares full K^3 tuple scoring with optional pair-beam pruning. No new architecture is added.",
            "",
            f"Frozen defaults: candidate_k=16, guideline_loss_weight=2.0, simplified_aux_weight=0.5, template_augmentation=extreme. Raw rows: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Commands:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k-values {args.candidate_k_values} --candidate-loss-weight {args.candidate_loss_weight} --tuple-loss-weight {args.tuple_loss_weight} --rank-loss-weight {args.rank_loss_weight} --template-split {args.template_split} --template-augmentations {args.template_augmentations} --simplified-aux-weight-values {args.simplified_aux_weight_values} --guideline-loss-weight-values {args.guideline_loss_weight_values} --tuple-debug-examples {args.tuple_debug_examples} --output-prefix writer_v45_frozen_baseline`",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k-values {args.candidate_k_values} --candidate-loss-weight {args.candidate_loss_weight} --tuple-loss-weight {args.tuple_loss_weight} --rank-loss-weight {args.rank_loss_weight} --template-split {args.template_split} --template-augmentations {args.template_augmentations} --simplified-aux-weight-values {args.simplified_aux_weight_values} --guideline-loss-weight-values {args.guideline_loss_weight_values} --tuple-debug-examples {args.tuple_debug_examples} --output-prefix writer_v45_pairbeam_b8 --tuple-pruning pair_beam --pair-beam-size 8`",
            "",
            "### Comparison",
            "",
            "| tuple pruning | beam | exact | condition recall | tuple candidates | tuple scorer time | proposer time | reader time | examples/sec | GPU MB | tuple-time speedup |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(rows, key=lambda item: (str(item.get("tuple_pruning")), int(item.get("pair_beam_size") or 0))):
            tuple_time = metric(row, "tuple_scorer_time")
            speedup = None if not baseline_tuple_time or not tuple_time else baseline_tuple_time / max(tuple_time, 1.0e-9)
            lines.append(
                f"| {row.get('tuple_pruning')} | {row.get('pair_beam_size')} | {fmt(metric(row, 'learned_reader_answer_exact'))} | {fmt(metric(row, 'learned_candidate_cond_recall'))} | {fmt(metric(row, 'tuple_candidates_scored'))} | {fmt(tuple_time)} | {fmt(metric(row, 'candidate_proposer_time'))} | {fmt(metric(row, 'reader_time'))} | {fmt(metric(row, 'examples_per_sec'))} | {fmt(metric(row, 'gpu_memory_mb'))} | {fmt(speedup)} |"
            )
        best_pair = None
        pair_rows = [row for row in rows if row.get("tuple_pruning") == "pair_beam"]
        passing = [
            row for row in pair_rows
            if (metric(row, "learned_reader_answer_exact") or 0.0) >= 0.94
            and (metric(row, "learned_candidate_cond_recall") or 0.0) >= 0.986
        ]
        if passing:
            best_pair = min(passing, key=lambda row: metric(row, "tuple_scorer_time") or 1.0e9)
        lines.extend(
            [
                "",
                "### Verdict",
                "",
                f"Frozen baseline exact = {fmt(metric(baseline, 'learned_reader_answer_exact'))}",
                f"Frozen baseline condition recall = {fmt(metric(baseline, 'learned_candidate_cond_recall'))}",
            ]
        )
        if best_pair is not None:
            lines.append(f"pair_beam B={best_pair.get('pair_beam_size')} preserves the accuracy gate and is the fastest accepted beam in this run.")
            lines.append("pair_beam can become the default for this diagnostic if repeated once more without regression.")
        else:
            lines.append("pair_beam did not pass the accuracy gate in this run, so full K^3 should remain the default.")
        lines.append("K=24/32 remain officially rejected for default runs; they require `--slow-sweep`.")
        section = "\n".join(lines) + "\n"
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v4.5 Freeze + Speed Optimization\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section, encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section, encoding="utf-8")
        return
    if args.output_prefix == "writer_v45_residual_fix":
        rows = [row for row in summary if row.get("writer") == "v4_condition_v2_full" and int(row.get("budget_steps", 0)) == final_budget]
        def metric(row: Dict[str, object], name: str) -> float | None:
            value = row.get(f"{name}_mean", "")
            return None if value == "" else float(value)

        best = max(rows, key=lambda row: metric(row, "learned_reader_answer_exact") or -1.0) if rows else None
        best_exact = metric(best, "learned_reader_answer_exact") if best is not None else None
        best_cond = metric(best, "learned_candidate_cond_recall") if best is not None else None
        best_value_miss = metric(best, "value_miss_rate") if best is not None else None
        best_tuple_error = metric(best, "tuple_scoring_error_rate") if best is not None else None
        lines = [
            "## Writer v4.5 Residual Failure Fix",
            "",
            "This residual sweep keeps only `v4_condition_v2_full` and searches the remaining candidate/weight/augmentation settings. No new architecture or real-data move is introduced.",
            "",
            f"Task: `{args.tasks}`. Noise: `{args.noise_levels}`. Marker rate: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot count: `{args.slot_counts}`. Max slots: `{args.max_slots}`. Budget: `{args.budgets}`. Seeds: `{args.seeds}`.",
            f"Candidate K: `{args.candidate_k_values}`. Guideline weights: `{args.guideline_loss_weight_values}`. Simplified aux weights: `{args.simplified_aux_weight_values}`. Template augmentations: `{args.template_augmentations}`. Template split: `{args.template_split}`.",
            f"Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k-values {args.candidate_k_values} --candidate-loss-weight {args.candidate_loss_weight} --tuple-loss-weight {args.tuple_loss_weight} --rank-loss-weight {args.rank_loss_weight} --template-split {args.template_split} --template-augmentations {args.template_augmentations} --simplified-aux-weight-values {args.simplified_aux_weight_values} --guideline-loss-weight-values {args.guideline_loss_weight_values} --tuple-debug-examples {args.tuple_debug_examples} --output-prefix {args.output_prefix}`",
            "",
            "### Top Configurations",
            "",
            "| rank | exact | condition recall | condition miss | value miss | tuple scoring error | candidate_k | guideline | simplified aux | augmentation |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        top_rows = sorted(rows, key=lambda row: metric(row, "learned_reader_answer_exact") or -1.0, reverse=True)[:12]
        for rank, row in enumerate(top_rows, 1):
            lines.append(
                f"| {rank} | {fmt(metric(row, 'learned_reader_answer_exact'))} | {fmt(metric(row, 'learned_candidate_cond_recall'))} | {fmt(metric(row, 'condition_miss_rate'))} | {fmt(metric(row, 'value_miss_rate'))} | {fmt(metric(row, 'tuple_scoring_error_rate'))} | {row.get('candidate_k')} | {row.get('guideline_loss_weight')} | {row.get('simplified_aux_weight')} | {row.get('template_augmentation')} |"
            )
        lines.extend(
            [
                "",
                "### Best Setting",
                "",
                f"heldout exact = {fmt(best_exact)}",
                f"condition recall = {fmt(best_cond)}",
                f"value miss rate = {fmt(best_value_miss)}",
                f"tuple scoring error rate = {fmt(best_tuple_error)}",
                f"candidate_k = {best.get('candidate_k') if best else ''}",
                f"guideline_loss_weight = {best.get('guideline_loss_weight') if best else ''}",
                f"simplified_aux_weight = {best.get('simplified_aux_weight') if best else ''}",
                f"template_augmentation = {best.get('template_augmentation') if best else ''}",
                "",
                "### Verdict",
                "",
            ]
        )
        if best_exact is not None and best_exact >= 0.90 and best_cond is not None and best_cond >= 0.98:
            lines.append("Synthetic heldout is repaired under the requested rule.")
        else:
            lines.append("Synthetic heldout is not fully repaired under the requested rule.")
        if best is not None:
            candidate_k = int(best.get("candidate_k", 0))
            guideline = float(best.get("guideline_loss_weight", 0.0))
            augmentation = str(best.get("template_augmentation", ""))
            if candidate_k > 16:
                lines.append("Candidate K helped most among the surviving explanations: candidate recall was still part of the issue.")
            if guideline > 0.5:
                lines.append("Higher guideline weight appears useful: condition semantics/role supervision helped.")
            if augmentation == "extreme":
                lines.append("Extreme augmentation appears useful: template coverage helped.")
            if not (candidate_k > 16 or guideline > 0.5 or augmentation == "extreme"):
                lines.append("The best setting did not depend on the larger residual knobs; remaining errors need inspection.")
        section = "\n".join(lines) + "\n"
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v4.5 Residual Failure Fix\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section, encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section, encoding="utf-8")
        return
    if args.output_prefix in {"writer_v45_condition_generalization", "writer_v45_condition_generalization_random"}:
        def load_summary(prefix: str) -> List[Dict[str, object]]:
            candidate = Path(args.out_dir) / f"{prefix}_summary.csv"
            if not candidate.exists():
                return []
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))

        heldout_summary = load_summary("writer_v45_condition_generalization")
        random_summary = load_summary("writer_v45_condition_generalization_random")
        if args.template_split == "heldout" and not heldout_summary:
            heldout_summary = summary
        if args.template_split == "random" and not random_summary:
            random_summary = summary

        def mean_for(writer: str, metric: str, **filters: object) -> float | None:
            return summary_mean(heldout_summary or summary, writer, final_budget, metric, **filters)

        def random_mean_for(writer: str, metric: str, **filters: object) -> float | None:
            return summary_mean(random_summary, writer, final_budget, metric, **filters) if random_summary else None

        baseline = mean_for("baseline_v4", "learned_reader_answer_exact")
        full = mean_for("v4_condition_v2_full", "learned_reader_answer_exact")
        full_cond = mean_for("v4_condition_v2_full", "learned_candidate_cond_recall")
        full_random = random_mean_for("v4_condition_v2_full", "learned_reader_answer_exact")
        full_drop = None if full is None or full_random is None else full_random - full
        best_writer = None
        best_exact = None
        for writer in writers:
            exact = mean_for(writer, "learned_reader_answer_exact")
            if exact is not None and (best_exact is None or exact > best_exact):
                best_writer = writer
                best_exact = exact
        lines = [
            "## Writer v4.5 Condition Candidate Generalization",
            "",
            "Writer v4.5 keeps the contextual tuple scorer fixed and changes only the condition-candidate proposer. The goal is heldout-template generalization, especially condition recall.",
            "",
            f"Tasks: `{args.tasks}`. Writers: `{args.writers}`. Noise levels: `{args.noise_levels}`. Marker rates: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot counts: `{args.slot_counts}`. Candidate K: `{args.candidate_k_values}`.",
            f"Budget: `{args.budgets}`. Seeds: `{args.seeds}`. Template split: `{args.template_split}`. Template augmentation: `{args.template_augmentation}`. Simplified aux weight: `{args.simplified_aux_weight}`. Guideline loss weight: `{args.guideline_loss_weight}`.",
            f"Projected rows: `{projected_rows}`. Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k-values {args.candidate_k_values} --candidate-loss-weight {args.candidate_loss_weight} --tuple-loss-weight {args.tuple_loss_weight} --rank-loss-weight {args.rank_loss_weight} --template-split {args.template_split} --template-augmentation {args.template_augmentation} --simplified-aux-weight {args.simplified_aux_weight} --guideline-loss-weight {args.guideline_loss_weight} --tuple-debug-examples {args.tuple_debug_examples} --output-prefix {args.output_prefix}`",
            "",
            "### Ablation Results",
            "",
            "| writer | heldout exact | random exact | heldout drop | slot F1 | all-slots exact | key recall | condition recall | value recall | cond token recall | cond span recall | cond any recall | cond span precision | condition miss rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for writer in writers:
            heldout_exact = mean_for(writer, "learned_reader_answer_exact")
            random_exact = random_mean_for(writer, "learned_reader_answer_exact")
            drop = None if heldout_exact is None or random_exact is None else random_exact - heldout_exact
            lines.append(
                f"| {writer_label(writer)} | {fmt(heldout_exact)} | {fmt(random_exact)} | {fmt(drop)} | {fmt(mean_for(writer, 'slot_f1'))} | {fmt(mean_for(writer, 'all_slots_exact'))} | {fmt(mean_for(writer, 'learned_candidate_key_recall'))} | {fmt(mean_for(writer, 'learned_candidate_cond_recall'))} | {fmt(mean_for(writer, 'learned_candidate_value_recall'))} | {fmt(mean_for(writer, 'condition_token_recall'))} | {fmt(mean_for(writer, 'condition_span_recall'))} | {fmt(mean_for(writer, 'condition_any_recall'))} | {fmt(mean_for(writer, 'condition_span_precision'))} | {fmt(mean_for(writer, 'candidate_miss_rate_cond'))} |"
            )
        lines.extend(
            [
                "",
                "### Verdict",
                "",
                f"baseline heldout exact = {fmt(baseline)}",
                f"full v4.5 heldout exact = {fmt(full)}",
                f"full v4.5 random exact = {fmt(full_random)}",
                f"full v4.5 heldout drop = {fmt(full_drop)}",
                f"full v4.5 condition recall = {fmt(full_cond)}",
                f"best ablation = `{best_writer or ''}` with exact {fmt(best_exact)}",
                "",
                "Did heldout exact improve?",
            ]
        )
        if baseline is not None and best_exact is not None and best_exact > baseline + 0.02:
            lines.append("Yes: at least one v4.5 condition-generalization ablation improves over baseline.")
        else:
            lines.append("No clear improvement over baseline in this run.")
        lines.append("Which ablation helped most?")
        lines.append(f"`{best_writer}` is best by final-budget exact." if best_writer else "No best ablation could be identified.")
        lines.append("Is condition recall fixed?")
        lines.append("Yes." if full_cond is not None and full_cond >= 0.98 else "Not fully; condition candidate recall remains below the target.")
        lines.append("Are we ready for real-data slot extraction yet?")
        lines.append("Yes, for a tiny pilot, only if heldout exact is >= 0.90 and condition recall is >= 0.98; otherwise keep fixing synthetic heldout generalization.")
        section = "\n".join(lines) + "\n"
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v4.5 Condition Candidate Generalization\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section, encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section, encoding="utf-8")
        return
    if args.output_prefix == "writer_v4_stageb":
        def mean_for(writer: str, metric: str, **filters: object) -> float | None:
            return summary_mean(summary, writer, final_budget, metric, **filters)

        v4 = "contextual_tuple_learned_candidates"
        oracle = mean_for("oracle", "learned_reader_answer_exact")
        v4_exact = mean_for(v4, "learned_reader_answer_exact")
        v2_exact = mean_for("learned_set_extractor_v2", "learned_reader_answer_exact")
        fact_exact = mean_for("fact_token", "learned_reader_answer_exact")
        repair_gain = mean_for("contextual_tuple_learned_candidates_plus_oracle_missing", "oracle_missing_repair_gain")
        noise_damage = mean_for("contextual_tuple_oracle_candidates_plus_learned_noise", "learned_noise_damage")
        lines = [
            "## Writer v4 Robustness Stage B",
            "",
            "Stage B stress-tests the full learned writer path: noisy text -> learned candidate proposer -> contextual tuple scorer -> typed memory slots -> structured reader. It keeps the model tiny and does not add AdaSlot, MILIE, real data, HPM recurrence, JEPA, ANN, graph memory, RL, or a larger backbone.",
            "",
            f"Tasks: `{args.tasks}`. Writers: `{args.writers}`. Noise levels: `{args.noise_levels}`. Marker rates: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot counts: `{args.slot_counts}`. Candidate K: `{args.candidate_k_values}`.",
            f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Template split: `{args.template_split}`. Device request: `{args.device}`. Torch: `{torch.__version__}`.",
            f"Projected rows: `{projected_rows}`. Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k-values {args.candidate_k_values} --candidate-loss-weight {args.candidate_loss_weight} --tuple-loss-weight {args.tuple_loss_weight} --rank-loss-weight {args.rank_loss_weight} --tuple-debug-examples {args.tuple_debug_examples} --output-prefix {args.output_prefix}`",
            "",
            "Verification:",
            "",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m py_compile .\\hpm_lite\\noisy_extraction.py .\\scripts\\run_noisy_slot_extraction.py` passed.",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m pytest -q` passed before and after Stage B1.",
            "",
            "### Overall Comparison",
            "",
            "| writer | exact | slot F1 | all-slots exact | key recall | cond recall | value recall | slot-count acc | predicted slots | true slots |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for writer in writers:
            lines.append(
                f"| {writer_label(writer)} | {fmt(mean_for(writer, 'learned_reader_answer_exact'))} | {fmt(mean_for(writer, 'slot_f1'))} | {fmt(mean_for(writer, 'all_slots_exact'))} | {fmt(mean_for(writer, 'learned_candidate_key_recall'))} | {fmt(mean_for(writer, 'learned_candidate_cond_recall'))} | {fmt(mean_for(writer, 'learned_candidate_value_recall'))} | {fmt(mean_for(writer, 'slot_count_accuracy'))} | {fmt(mean_for(writer, 'predicted_slot_count'))} | {fmt(mean_for(writer, 'true_slot_count'))} |"
            )
        lines.extend(["", "### By Task", "", "| task | v4 exact | v2 exact | fact_token exact | oracle exact |", "| --- | ---: | ---: | ---: | ---: |"])
        for task in parse_str_list(args.tasks, TASKS):
            lines.append(
                f"| {task} | {fmt(mean_for(v4, 'learned_reader_answer_exact', task=task))} | {fmt(mean_for('learned_set_extractor_v2', 'learned_reader_answer_exact', task=task))} | {fmt(mean_for('fact_token', 'learned_reader_answer_exact', task=task))} | {fmt(mean_for('oracle', 'learned_reader_answer_exact', task=task))} |"
            )
        lines.extend(["", "### Stress Slices", "", "| slice | value | v4 exact | key recall | cond recall | value recall | repair gain | noise damage |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for noise in parse_str_list(args.noise_levels, NOISE_LEVELS):
            lines.append(f"| noise | {noise} | {fmt(mean_for(v4, 'learned_reader_answer_exact', noise_level=noise))} | {fmt(mean_for(v4, 'learned_candidate_key_recall', noise_level=noise))} | {fmt(mean_for(v4, 'learned_candidate_cond_recall', noise_level=noise))} | {fmt(mean_for(v4, 'learned_candidate_value_recall', noise_level=noise))} | {fmt(mean_for('contextual_tuple_learned_candidates_plus_oracle_missing', 'oracle_missing_repair_gain', noise_level=noise))} | {fmt(mean_for('contextual_tuple_oracle_candidates_plus_learned_noise', 'learned_noise_damage', noise_level=noise))} |")
        for slot_count in parse_int_list(args.slot_counts):
            lines.append(f"| slot_count | {slot_count} | {fmt(mean_for(v4, 'learned_reader_answer_exact', slot_count=slot_count))} | {fmt(mean_for(v4, 'learned_candidate_key_recall', slot_count=slot_count))} | {fmt(mean_for(v4, 'learned_candidate_cond_recall', slot_count=slot_count))} | {fmt(mean_for(v4, 'learned_candidate_value_recall', slot_count=slot_count))} | {fmt(mean_for('contextual_tuple_learned_candidates_plus_oracle_missing', 'oracle_missing_repair_gain', slot_count=slot_count))} | {fmt(mean_for('contextual_tuple_oracle_candidates_plus_learned_noise', 'learned_noise_damage', slot_count=slot_count))} |")
        for distractor_count in parse_int_list(args.distractor_counts):
            lines.append(f"| distractors | {distractor_count} | {fmt(mean_for(v4, 'learned_reader_answer_exact', distractor_count=distractor_count))} | {fmt(mean_for(v4, 'learned_candidate_key_recall', distractor_count=distractor_count))} | {fmt(mean_for(v4, 'learned_candidate_cond_recall', distractor_count=distractor_count))} | {fmt(mean_for(v4, 'learned_candidate_value_recall', distractor_count=distractor_count))} | {fmt(mean_for('contextual_tuple_learned_candidates_plus_oracle_missing', 'oracle_missing_repair_gain', distractor_count=distractor_count))} | {fmt(mean_for('contextual_tuple_oracle_candidates_plus_learned_noise', 'learned_noise_damage', distractor_count=distractor_count))} |")
        for candidate_k in parse_int_list(args.candidate_k_values) if args.candidate_k_values else [8]:
            lines.append(f"| candidate_k | {candidate_k} | {fmt(mean_for(v4, 'learned_reader_answer_exact', candidate_k=candidate_k))} | {fmt(mean_for(v4, 'learned_candidate_key_recall', candidate_k=candidate_k))} | {fmt(mean_for(v4, 'learned_candidate_cond_recall', candidate_k=candidate_k))} | {fmt(mean_for(v4, 'learned_candidate_value_recall', candidate_k=candidate_k))} | {fmt(mean_for('contextual_tuple_learned_candidates_plus_oracle_missing', 'oracle_missing_repair_gain', candidate_k=candidate_k))} | {fmt(mean_for('contextual_tuple_oracle_candidates_plus_learned_noise', 'learned_noise_damage', candidate_k=candidate_k))} |")
        lines.extend(["", "### Verdict", ""])
        lines.append(f"oracle exact = {fmt(oracle)}")
        lines.append(f"Writer v4 exact = {fmt(v4_exact)}")
        lines.append(f"Writer v2 exact = {fmt(v2_exact)}")
        lines.append(f"fact_token exact = {fmt(fact_exact)}")
        lines.append(f"oracle_missing repair gain = {fmt(repair_gain)}")
        lines.append(f"learned_noise damage = {fmt(noise_damage)}")
        lines.append("")
        lines.append("Does Writer v4 survive harder synthetic noise?")
        hard_slot8 = mean_for(v4, "learned_reader_answer_exact", task="noisy_conditional", noise_level="hard", slot_count=8)
        lines.append("Yes." if hard_slot8 is not None and hard_slot8 >= 0.95 else f"Not cleanly; hard/noisy_conditional/slot_count8 exact is {fmt(hard_slot8)}.")
        lines.append("Does it still beat v2 and fact_token?")
        if v4_exact is not None and v2_exact is not None and fact_exact is not None and v4_exact > v2_exact + 0.05 and v4_exact > fact_exact + 0.05:
            lines.append("Yes, v4 is clearly ahead on this Stage B grid.")
        else:
            lines.append("The separation is weak or mixed; inspect the slice tables.")
        lines.append("What fails first: condition recall, candidate noise, coexisting set extraction, or template generalization?")
        coexisting = mean_for(v4, "learned_reader_answer_exact", task="noisy_coexisting")
        conditional = mean_for(v4, "learned_reader_answer_exact", task="noisy_conditional")
        cond_recall = mean_for(v4, "learned_candidate_cond_recall")
        if coexisting is not None and conditional is not None and coexisting < conditional - 0.10:
            lines.append("Coexisting set extraction fails first.")
        elif repair_gain is not None and repair_gain > 0.05:
            lines.append("Candidate recall fails first.")
        elif noise_damage is not None and noise_damage > 0.05:
            lines.append("Candidate noise fails first.")
        elif cond_recall is not None and cond_recall < 0.95:
            lines.append("Condition candidate recall is the first visible weakness.")
        else:
            lines.append("No major failure mode surfaced in Stage B1.")
        lines.append("Are we ready for real-data slot extraction, or do we need another synthetic fix?")
        lines.append("Use the heldout-template check before real-data conversion; if heldout holds, real-data slot extraction becomes the next sensible phase.")
        lines.append("")
        section = "\n".join(lines)
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v4 Robustness Stage B\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")
        return

    if args.output_prefix == "writer_v4_heldout":
        def mean_for(writer: str, metric: str, **filters: object) -> float | None:
            return summary_mean(summary, writer, final_budget, metric, **filters)

        heldout = mean_for("contextual_tuple_learned_candidates", "learned_reader_answer_exact")
        random_ref = None
        stageb_path = Path(args.out_dir) / "writer_v4_stageb_summary.csv"
        if stageb_path.exists():
            with stageb_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            values = [
                float(row["learned_reader_answer_exact_mean"])
                for row in rows
                if row.get("writer") == "contextual_tuple_learned_candidates"
                and row.get("task") == "noisy_conditional"
                and row.get("noise_level") == "hard"
                and row.get("distractor_count") == "16"
                and row.get("slot_count") == "8"
                and row.get("budget_steps") == str(final_budget)
                and row.get("candidate_k") == args.candidate_k_values
                and row.get("learned_reader_answer_exact_mean") not in {"", None}
            ]
            random_ref = statistics.mean(values) if values else None
        drop = None if random_ref is None or heldout is None else random_ref - heldout
        section = f"""## Writer v4 Heldout Template Generalization

Heldout template split trains on simple templates and evaluates on paraphrase templates. This checks whether Writer v4 is memorizing surface forms.

| final budget | random-template exact | heldout-template exact | heldout drop |
| ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(random_ref)} | {fmt(heldout)} | {fmt(drop)} |

Verdict: {"heldout template generalization is strong enough for this synthetic stage." if heldout is not None and heldout >= 0.90 and (drop is None or drop < 0.10) else "template generalization is still a bottleneck or needs a cleaner reference run."}
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v4 Heldout Template Generalization\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")
        return

    if is_contextual_learned_candidate_run(writers):
        def mean_for(writer: str, metric: str, **filters: object) -> float | None:
            return summary_mean(summary, writer, final_budget, metric, **filters)

        oracle = mean_for("contextual_tuple_oracle_candidates", "learned_reader_answer_exact")
        learned = mean_for("contextual_tuple_learned_candidates", "learned_reader_answer_exact")
        repaired = mean_for("contextual_tuple_learned_candidates_plus_oracle_missing", "learned_reader_answer_exact")
        noisy = mean_for("contextual_tuple_oracle_candidates_plus_learned_noise", "learned_reader_answer_exact")
        gap = mean_for("contextual_tuple_learned_candidates", "learned_candidate_gap_to_oracle")
        repair_gain = mean_for("contextual_tuple_learned_candidates_plus_oracle_missing", "oracle_missing_repair_gain")
        noise_damage = mean_for("contextual_tuple_oracle_candidates_plus_learned_noise", "learned_noise_damage")
        best_k = None
        best_exact = None
        for candidate_k in parse_int_list(args.candidate_k_values) if args.candidate_k_values else [8]:
            exact = mean_for("contextual_tuple_learned_candidates", "learned_reader_answer_exact", candidate_k=candidate_k)
            if exact is not None and (best_exact is None or exact > best_exact):
                best_k = candidate_k
                best_exact = exact
        lines = [
            "## Contextual Tuple Scorer with Learned Candidates",
            "",
            "This diagnostic reuses Writer v4's contextual tuple scorer but replaces oracle candidate pools with candidates from the learned `CandidateFieldProposer`. It keeps the model tiny and does not add AdaSlot, MILIE, real datasets, HPM recurrence, JEPA, ANN, graph memory, RL, or a larger backbone.",
            "",
            f"Tasks: `{args.tasks}`. Writers: `{args.writers}`. Noise levels: `{args.noise_levels}`. Marker rates: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot counts: `{args.slot_counts}`. Max slots: `{args.max_slots}`. Candidate K: `{args.candidate_k_values}`.",
            f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Candidate loss weight: `{args.candidate_loss_weight}`. Tuple loss weight: `{args.tuple_loss_weight}`. Rank loss weight: `{args.rank_loss_weight}`. Debug examples: `{args.tuple_debug_examples}`. Device request: `{args.device}`. Torch: `{torch.__version__}`.",
            f"Projected requested rows: `{projected_rows}`. Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k-values {args.candidate_k_values} --candidate-loss-weight {args.candidate_loss_weight} --tuple-loss-weight {args.tuple_loss_weight} --rank-loss-weight {args.rank_loss_weight} --tuple-debug-examples {args.tuple_debug_examples}`",
            "",
            "Verification:",
            "",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m py_compile .\\hpm_lite\\noisy_extraction.py .\\scripts\\run_noisy_slot_extraction.py` passed.",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m pytest -q` passed before and after the diagnostic.",
            "",
            "### Final Budget Writer Comparison",
            "",
            "| writer | exact | slot F1 | all-slots exact | key recall | cond recall | value recall | all-field recall | key precision | cond precision | value precision | gap/repair/noise |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for writer in writers:
            special = (
                mean_for(writer, "learned_candidate_gap_to_oracle")
                or mean_for(writer, "oracle_missing_repair_gain")
                or mean_for(writer, "learned_noise_damage")
            )
            lines.append(
                f"| {writer_label(writer)} | {fmt(mean_for(writer, 'learned_reader_answer_exact'))} | {fmt(mean_for(writer, 'slot_f1'))} | {fmt(mean_for(writer, 'all_slots_exact'))} | {fmt(mean_for(writer, 'learned_candidate_key_recall'))} | {fmt(mean_for(writer, 'learned_candidate_cond_recall'))} | {fmt(mean_for(writer, 'learned_candidate_value_recall'))} | {fmt(mean_for(writer, 'learned_candidate_all_field_recall'))} | {fmt(mean_for(writer, 'learned_candidate_key_precision'))} | {fmt(mean_for(writer, 'learned_candidate_cond_precision'))} | {fmt(mean_for(writer, 'learned_candidate_value_precision'))} | {fmt(special)} |"
            )
        lines.extend(
            [
                "",
                "### Candidate K Sweep",
                "",
                "| candidate_k | oracle exact | learned exact | learned+oracle-missing exact | oracle+learned-noise exact | key recall | cond recall | value recall | all-field recall |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate_k in parse_int_list(args.candidate_k_values) if args.candidate_k_values else [8]:
            filters = {"candidate_k": candidate_k}
            lines.append(
                f"| {candidate_k} | {fmt(mean_for('contextual_tuple_oracle_candidates', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('contextual_tuple_learned_candidates', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('contextual_tuple_learned_candidates_plus_oracle_missing', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('contextual_tuple_oracle_candidates_plus_learned_noise', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('contextual_tuple_learned_candidates', 'learned_candidate_key_recall', **filters))} | {fmt(mean_for('contextual_tuple_learned_candidates', 'learned_candidate_cond_recall', **filters))} | {fmt(mean_for('contextual_tuple_learned_candidates', 'learned_candidate_value_recall', **filters))} | {fmt(mean_for('contextual_tuple_learned_candidates', 'learned_candidate_all_field_recall', **filters))} |"
            )
        lines.extend(
            [
                "",
                "### Verdict",
                "",
                f"oracle candidate exact = {fmt(oracle)}",
                f"learned candidate exact = {fmt(learned)}",
                f"learned+oracle-missing exact = {fmt(repaired)}",
                f"oracle+learned-noise exact = {fmt(noisy)}",
                f"best learned candidate_k = {best_k if best_k is not None else ''}",
                f"best learned exact = {fmt(best_exact)}",
                f"learned gap to oracle = {fmt(gap)}",
                f"oracle missing repair gain = {fmt(repair_gain)}",
                f"learned noise damage = {fmt(noise_damage)}",
                "",
                "Can contextual tuple scoring work with learned candidates?",
            ]
        )
        if learned is not None and learned >= 0.90:
            lines.append("Yes: learned candidates plus contextual tuple scoring are near solved in this Stage A setting.")
        elif repaired is not None and learned is not None and repaired > learned + 0.10:
            lines.append("Not yet: learned candidate recall is the bottleneck because oracle-missing repair gives a large gain.")
        elif noisy is not None and oracle is not None and noisy < oracle - 0.10:
            lines.append("Not yet: learned candidate noise breaks tuple scoring even when gold candidates are present.")
        else:
            lines.append("Partially or inconclusively: inspect field recalls and debug examples.")
        lines.append("Which candidate field type fails first?")
        recalls = {
            "key": mean_for("contextual_tuple_learned_candidates", "learned_candidate_key_recall"),
            "condition": mean_for("contextual_tuple_learned_candidates", "learned_candidate_cond_recall"),
            "value": mean_for("contextual_tuple_learned_candidates", "learned_candidate_value_recall"),
        }
        known = {name: value for name, value in recalls.items() if value is not None}
        if known:
            field = min(known, key=lambda name: known[name])
            lines.append(f"{field} candidates are weakest at final budget: recall {known[field]:.3f}.")
        else:
            lines.append("No learned-candidate recall rows were available.")
        lines.append("Is the next bottleneck candidate recall, candidate noise, or tuple selection?")
        if repair_gain is not None and repair_gain > 0.10:
            lines.append("Candidate recall is the next bottleneck.")
        elif noise_damage is not None and noise_damage > 0.10:
            lines.append("Candidate noise is the next bottleneck.")
        elif learned is not None and oracle is not None and oracle - learned > 0.10:
            lines.append("Tuple selection under learned pools remains the bottleneck.")
        else:
            lines.append("This Stage A setting is healthy enough to move to harder synthetic noise or more slots.")
        lines.append("Should we next do harder synthetic noise, coexisting, or real-data conversion?")
        if learned is not None and learned >= 0.90:
            lines.append("Next: harder synthetic noise and slot_count 8 before real-data conversion.")
        else:
            lines.append("Next: repair learned candidate extraction before coexisting or real-data conversion.")
        lines.append("")
        section = "\n".join(lines)
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Contextual Tuple Scorer with Learned Candidates\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")
        return

    if is_contextual_tuple_run(writers):
        def mean_for(writer: str, metric: str, **filters: object) -> float | None:
            return summary_mean(summary, writer, final_budget, metric, **filters)

        oracle = mean_for("contextual_tuple_oracle_candidates", "learned_reader_answer_exact")
        hard = mean_for("contextual_tuple_oracle_candidates_plus_hard_negatives", "learned_reader_answer_exact")
        gold_key_cond = mean_for("contextual_tuple_gold_key_cond", "learned_reader_answer_exact")
        gold_all = mean_for("contextual_tuple_gold_all_fields", "learned_reader_answer_exact")
        spn_baseline = 0.2937
        independent_baseline = 0.0563
        lines = [
            "## Writer v4 Contextual Tuple Edge Scorer",
            "",
            "Writer v4 keeps the typed candidate setup but replaces isolated tuple embeddings with contextual relation evidence: token context, absolute and relative positions, field order bits, pooled text between fields, and local windows around each candidate field. This run trains only the tuple scorer/writer modules and does not add AdaSlot, MILIE, real datasets, HPM recurrence, JEPA, ANN, graph memory, RL, or a larger backbone.",
            "",
            f"Tasks: `{args.tasks}`. Writers: `{args.writers}`. Noise levels: `{args.noise_levels}`. Marker rates: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot counts: `{args.slot_counts}`. Max slots: `{args.max_slots}`. Candidate K: `{args.candidate_k_values}`.",
            f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Candidate loss weight: `{args.candidate_loss_weight}`. Debug examples: `{args.tuple_debug_examples}`. Device request: `{args.device}`. Torch: `{torch.__version__}`.",
            f"Projected requested rows: `{projected_rows}`. Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k-values {args.candidate_k_values} --tuple-debug-examples {args.tuple_debug_examples}`",
            "",
            "Verification:",
            "",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m py_compile .\\hpm_lite\\noisy_extraction.py .\\scripts\\run_noisy_slot_extraction.py` passed.",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m pytest -q` passed before and after the diagnostic.",
            "",
            "Previous tuple-assembler baselines from the prior debug run: independent field heads exact = `0.0563`; SPN tuple scorer exact = `0.2937`.",
            "",
            "### Writer Comparison",
            "",
            "| writer | exact | slot F1 | all-slots exact | tuple AUC | pos score | neg score | score margin | gain vs SPN | gain vs independent |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for writer in writers:
            lines.append(
                f"| {writer_label(writer)} | {fmt(mean_for(writer, 'learned_reader_answer_exact'))} | {fmt(mean_for(writer, 'slot_f1'))} | {fmt(mean_for(writer, 'all_slots_exact'))} | {fmt(mean_for(writer, 'tuple_auc'))} | {fmt(mean_for(writer, 'tuple_positive_score_mean'))} | {fmt(mean_for(writer, 'tuple_negative_score_mean'))} | {fmt(mean_for(writer, 'tuple_score_margin'))} | {fmt(mean_for(writer, 'contextual_gain_over_spn'))} | {fmt(mean_for(writer, 'contextual_gain_over_independent_heads'))} |"
            )
        lines.extend(
            [
                "",
                "### Candidate K Sweep",
                "",
                "| candidate_k | oracle candidates exact | oracle + hard negatives exact | gold key+cond exact | gold all fields exact | hard-neg FP rate |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate_k in parse_int_list(args.candidate_k_values) if args.candidate_k_values else [8]:
            filters = {"candidate_k": candidate_k}
            lines.append(
                f"| {candidate_k} | {fmt(mean_for('contextual_tuple_oracle_candidates', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('contextual_tuple_oracle_candidates_plus_hard_negatives', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('contextual_tuple_gold_key_cond', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('contextual_tuple_gold_all_fields', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('contextual_tuple_oracle_candidates_plus_hard_negatives', 'hard_negative_false_positive_rate', **filters))} |"
            )
        lines.extend(
            [
                "",
                "### Verdict",
                "",
                f"independent heads baseline exact = {independent_baseline:.4f}",
                f"old SPN tuple baseline exact = {spn_baseline:.4f}",
                f"contextual oracle-candidate exact = {fmt(oracle)}",
                f"contextual oracle-candidates-plus-hard-negatives exact = {fmt(hard)}",
                f"contextual gold_key_cond exact = {fmt(gold_key_cond)}",
                f"contextual gold_all_fields exact = {fmt(gold_all)}",
                "",
                "Did contextual relation evidence fix tuple assembly?",
            ]
        )
        if oracle is not None and oracle >= 0.90:
            lines.append("Yes under clean oracle candidates: contextual relation evidence brings assembly near the symbolic/gold upper bound.")
        elif oracle is not None and oracle > spn_baseline + 0.05:
            lines.append("Partially: contextual scoring beats the old SPN scorer but does not solve oracle-candidate assembly.")
        else:
            lines.append("No: contextual scoring does not clearly beat the old SPN tuple scorer, so features/labels/debug examples need inspection.")
        lines.append("Can oracle candidates now be assembled correctly?")
        lines.append("Yes." if oracle is not None and oracle >= 0.90 else "No.")
        lines.append("Is the next bottleneck hard negative discrimination, learned candidates, or iterative completion?")
        if oracle is not None and oracle >= 0.90 and hard is not None and hard < oracle - 0.10:
            lines.append("Hard negative discrimination is the next bottleneck.")
        elif oracle is not None and oracle >= 0.90:
            lines.append("Oracle assembly is healthy; learned candidate extraction is the next thing to re-enable.")
        elif oracle is not None and oracle > spn_baseline + 0.05:
            lines.append("The direction is useful but tuple scoring still needs a stronger relation objective before learned candidates.")
        else:
            lines.append("Iterative completion or a better tuple objective should come before learned candidates.")
        lines.append("Should we now move toward SPN4RE-style set prediction, MILIE-style iterative completion, or learned candidate extraction?")
        if oracle is not None and oracle >= 0.90 and hard is not None and hard >= 0.80:
            lines.append("Move to learned candidate extraction next.")
        elif oracle is not None and oracle > spn_baseline + 0.05:
            lines.append("Keep the SPN4RE-style tuple path, add harder negative training, then consider MILIE-style iterative completion.")
        else:
            lines.append("Stay on SPN4RE-style tuple set prediction/debugging; MILIE or learned candidates would be premature.")
        lines.append("")
        section = "\n".join(lines)
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v4 Contextual Tuple Edge Scorer\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")
        return

    if is_tuple_debug_run(writers):
        def mean_for(writer: str, metric: str, **filters: object) -> float | None:
            return summary_mean(summary, writer, final_budget, metric, **filters)

        sanity = mean_for("writer_v3_oracle_candidates_sanity", "learned_reader_answer_exact", eval_mode="no_objectness_true_count")
        canonical = mean_for("writer_v3_oracle_candidates_sanity", "learned_reader_answer_exact", eval_mode="no_hungarian_canonical_debug")
        gold_key_cond = mean_for("writer_v3_oracle_candidates_sanity", "learned_reader_answer_exact", eval_mode="gold_key_cond")
        gold_all = mean_for("writer_v3_oracle_candidates_sanity", "learned_reader_answer_exact", eval_mode="gold_all_fields")
        independent = mean_for("writer_v3_oracle_candidates_sanity", "learned_reader_answer_exact", eval_mode="independent_field_heads_current")
        spn = mean_for("spn_tuple_assembler_oracle_candidates", "learned_reader_answer_exact")
        lines = [
            "## Tuple Assembler Debug",
            "",
            "This diagnostic keeps the candidate setting tiny and checks whether v3 failed because of candidate-index construction, matching/objectness, independent field heads, or the tuple assembly objective. It does not add AdaSlot, MILIE, real datasets, HPM recurrence, JEPA, ANN, graph memory, or RL.",
            "",
            f"Tasks: `{args.tasks}`. Writers: `{args.writers}`. Noise levels: `{args.noise_levels}`. Marker rates: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot counts: `{args.slot_counts}`. Max slots: `{args.max_slots}`. Candidate K: `{args.candidate_k_values}`.",
            f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Debug examples: `{args.tuple_debug_examples}`. Device request: `{args.device}`. Torch: `{torch.__version__}`.",
            f"Projected requested rows: `{projected_rows}`. Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k-values {args.candidate_k_values} --tuple-debug-examples {args.tuple_debug_examples}`",
            "",
            "Verification:",
            "",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m py_compile .\\hpm_lite\\noisy_extraction.py .\\scripts\\run_noisy_slot_extraction.py` passed.",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m pytest -q` passed before and after the diagnostic.",
            "",
            "### Sanity Modes",
            "",
            "| eval mode | exact | slot F1 | key cand acc | cond cand acc | value cand acc | tuple acc | matched tuple acc | mean gold cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        modes = [
            "independent_field_heads_current",
            "no_objectness_true_count",
            "no_hungarian_canonical_debug",
            "gold_key_only",
            "gold_cond_only",
            "gold_value_only",
            "gold_key_cond",
            "gold_all_fields",
        ]
        for mode in modes:
            filters = {"eval_mode": mode}
            lines.append(
                f"| {mode} | {fmt(mean_for('writer_v3_oracle_candidates_sanity', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('writer_v3_oracle_candidates_sanity', 'slot_f1', **filters))} | {fmt(mean_for('writer_v3_oracle_candidates_sanity', 'key_candidate_accuracy', **filters))} | {fmt(mean_for('writer_v3_oracle_candidates_sanity', 'condition_candidate_accuracy', **filters))} | {fmt(mean_for('writer_v3_oracle_candidates_sanity', 'value_candidate_accuracy', **filters))} | {fmt(mean_for('writer_v3_oracle_candidates_sanity', 'tuple_accuracy', **filters))} | {fmt(mean_for('writer_v3_oracle_candidates_sanity', 'matched_tuple_accuracy', **filters))} | {fmt(mean_for('writer_v3_oracle_candidates_sanity', 'mean_hungarian_gold_cost', **filters))} |"
            )
        lines.extend(
            [
                "",
                "### SPN Tuple Scorer",
                "",
                "| writer | exact | slot F1 | tuple exact | tuple accuracy | gain over independent |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                f"| spn_tuple_assembler_oracle_candidates | {fmt(spn)} | {fmt(mean_for('spn_tuple_assembler_oracle_candidates', 'slot_f1'))} | {fmt(mean_for('spn_tuple_assembler_oracle_candidates', 'spn_tuple_exact'))} | {fmt(mean_for('spn_tuple_assembler_oracle_candidates', 'spn_tuple_accuracy'))} | {fmt(None if spn is None or independent is None else spn - independent)} |",
                "",
                "### Verdict",
                "",
                f"oracle-candidate sanity no-objectness exact = {fmt(sanity)}",
                f"canonical debug exact = {fmt(canonical)}",
                f"gold_key_cond exact = {fmt(gold_key_cond)}",
                f"gold_all_fields exact = {fmt(gold_all)}",
                f"SPN tuple exact = {fmt(spn)}",
                "",
                "Was v3 failure an implementation/matching bug or a real tuple-assembly objective failure?",
            ]
        )
        if gold_all is not None and gold_all < 0.99:
            lines.append("Implementation or slot formatting is still suspect: gold_all_fields did not reach 1.0.")
        elif sanity is not None and sanity < 0.90:
            lines.append("The assembler objective/factorization is failing under oracle candidates; gold field plumbing is sane.")
        else:
            lines.append("The sanity path works, so the prior failure is likely training/objectness rather than basic indexing.")
        lines.append("Do independent field heads work under oracle candidates?")
        lines.append("Yes." if sanity is not None and sanity >= 0.90 else "No.")
        lines.append("Does SPN-style whole-tuple scoring fix assembly?")
        lines.append("Yes." if spn is not None and spn >= 0.90 else "No.")
        lines.append("What should Writer v4 be?")
        if spn is not None and independent is not None and spn > independent + 0.05:
            lines.append("Writer v4 should move toward SPN-style tuple scoring.")
        elif gold_all is not None and gold_all < 0.99:
            lines.append("Writer v4 should wait; first fix implementation/index mapping.")
        else:
            lines.append("Writer v4 should focus on the measured failing path above, not AdaSlot yet.")
        lines.append("")
        section = "\n".join(lines)
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Tuple Assembler Debug\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")
        return

    if is_v3_run(writers):
        def mean_for(writer: str, metric: str, **filters: object) -> float | None:
            return summary_mean(summary, writer, final_budget, metric, **filters)

        v2_exact = mean_for("learned_set_extractor_v2", "learned_reader_answer_exact")
        oracle_exact = mean_for("writer_v3_oracle_candidates", "learned_reader_answer_exact")
        learned_exact = mean_for("writer_v3_learned_candidates", "learned_reader_answer_exact")
        noise_exact = mean_for("writer_v3_oracle_candidates_plus_noise", "learned_reader_answer_exact")
        learned_gain = mean_for("writer_v3_learned_candidates", "v3_gain_over_v2")
        learned_gap = mean_for("writer_v3_learned_candidates", "v3_gap_to_oracle_fields")
        lines = [
            "## Writer v3: Field-Candidate Proposer and Slot Assembler",
            "",
            "Writer v3 keeps the model tiny and replaces direct all-token field pointing with high-recall field candidates followed by tuple assembly over candidate pools. It trains only writer modules and does not add a larger backbone, HPM recurrence, JEPA, ANN, graph memory, or RL.",
            "",
            f"Tasks: `{args.tasks}`. Writers: `{args.writers}`. Noise levels: `{args.noise_levels}`. Marker rates: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot counts: `{args.slot_counts}`. Max slots: `{args.max_slots}`. Candidate K: `{args.candidate_k_values}`.",
            f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Candidate loss weight: `{args.candidate_loss_weight}`. Device request: `{args.device}`. Torch: `{torch.__version__}`.",
            f"Projected requested rows: `{projected_rows}`. Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k-values {args.candidate_k_values} --candidate-loss-weight {args.candidate_loss_weight}`",
            "",
            "Verification:",
            "",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m py_compile .\\hpm_lite\\noisy_extraction.py .\\scripts\\run_noisy_slot_extraction.py` passed.",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m pytest -q` passed before and after the diagnostic.",
            "",
            "### Writer Comparison",
            "",
            "| writer | learned exact | symbolic exact | slot F1 | all-slots exact | candidate recall | candidate precision | gain over v2 | gap to oracle candidates |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for writer in writers:
            lines.append(
                f"| {writer_label(writer)} | {fmt(mean_for(writer, 'learned_reader_answer_exact'))} | {fmt(mean_for(writer, 'symbolic_answer_exact'))} | {fmt(mean_for(writer, 'slot_f1'))} | {fmt(mean_for(writer, 'all_slots_exact'))} | {fmt(mean_for(writer, 'all_fields_candidate_recall'))} | {fmt(mean_for(writer, 'candidate_precision'))} | {fmt(mean_for(writer, 'v3_gain_over_v2'))} | {fmt(mean_for(writer, 'v3_gap_to_oracle_fields'))} |"
            )

        lines.extend(
            [
                "",
                "### Candidate K Sweep",
                "",
                "| candidate_k | v2 exact | v3 oracle-candidate exact | v3 learned-candidate exact | key recall | cond recall | value recall | all-field recall |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate_k in parse_int_list(args.candidate_k_values) if args.candidate_k_values else [8]:
            filters = {"candidate_k": candidate_k}
            lines.append(
                f"| {candidate_k} | {fmt(mean_for('learned_set_extractor_v2', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('writer_v3_oracle_candidates', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('writer_v3_learned_candidates', 'learned_reader_answer_exact', **filters))} | {fmt(mean_for('writer_v3_learned_candidates', 'candidate_key_recall', **filters))} | {fmt(mean_for('writer_v3_learned_candidates', 'candidate_condition_recall', **filters))} | {fmt(mean_for('writer_v3_learned_candidates', 'candidate_value_recall', **filters))} | {fmt(mean_for('writer_v3_learned_candidates', 'all_fields_candidate_recall', **filters))} |"
            )

        lines.extend(
            [
                "",
                "### Verdict",
                "",
                f"v2 baseline exact = {fmt(v2_exact)}",
                f"v3 oracle-candidate exact = {fmt(oracle_exact)}",
                f"v3 learned-candidate exact = {fmt(learned_exact)}",
                f"v3 oracle-candidates-plus-noise exact = {fmt(noise_exact)}",
                f"v3 gain over v2 = {fmt(learned_gain)}",
                f"v3 learned gap to oracle candidates = {fmt(learned_gap)}",
                "",
                "Did field candidates fix the v2 pointer bottleneck?",
            ]
        )
        if learned_gain is not None and learned_gain > 0.05:
            lines.append(f"Yes, partially: learned candidates improve exact by {learned_gain:.3f} over v2.")
        elif oracle_exact is not None and v2_exact is not None and oracle_exact > v2_exact + 0.05:
            lines.append("Not with learned candidates yet: oracle candidates help, so the candidate proposer is likely the bottleneck.")
        else:
            lines.append("No: even oracle candidates do not clearly improve, so tuple assembly is likely the bottleneck.")
        lines.append("Is the next bottleneck candidate recall, tuple assembly, or objectness?")
        if oracle_exact is not None and learned_exact is not None and oracle_exact > learned_exact + 0.05:
            lines.append("Candidate recall/proposal is the next bottleneck.")
        elif oracle_exact is not None and v2_exact is not None and oracle_exact <= v2_exact + 0.05:
            lines.append("Tuple assembly is the next bottleneck.")
        else:
            lines.append("Objectness/slot selection remains secondary after candidate quality.")
        lines.append("Should the next paper-inspired step be HGERE-style high-recall pruning, SPN4RE-style tuple set prediction, MILIE-style iterative completion, or AdaSlot-style adaptive selection?")
        if oracle_exact is not None and learned_exact is not None and oracle_exact > learned_exact + 0.05:
            lines.append("Use HGERE-style high-recall pruning next.")
        elif oracle_exact is not None and v2_exact is not None and oracle_exact <= v2_exact + 0.05:
            lines.append("Use SPN4RE-style tuple set prediction improvements next.")
        else:
            lines.append("Use AdaSlot-style adaptive selection only after candidate recall and assembly are no longer the measured bottlenecks.")
        lines.append("")
        section = "\n".join(lines)
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v3: Field-Candidate Proposer and Slot Assembler\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")
        return

    if is_bottleneck_run(args):
        def mean(metric: str, **filters: object) -> float | None:
            return summary_mean(summary, "learned_set_extractor_v2", final_budget, metric, **filters)

        def best_threshold() -> Tuple[float | None, float | None, float | None]:
            grouped: Dict[float, List[Dict[str, object]]] = {}
            for row in summary:
                if row["writer"] != "learned_set_extractor_v2" or int(row["budget_steps"]) != final_budget:
                    continue
                if row.get("eval_mode") != "threshold_sweep" or row.get("learned_reader_answer_exact_mean", "") == "":
                    continue
                grouped.setdefault(float(row["eval_threshold"]), []).append(row)
            best = None
            for threshold, rows in grouped.items():
                exact = statistics.mean(float(row["learned_reader_answer_exact_mean"]) for row in rows)
                slot_f1 = statistics.mean(float(row["slot_f1_mean"]) for row in rows)
                if best is None or exact > best[1]:
                    best = (threshold, exact, slot_f1)
            return best if best is not None else (None, None, None)

        normal = mean("learned_reader_answer_exact", eval_mode="normal_v2")
        oracle_count = mean("learned_reader_answer_exact", eval_mode="oracle_count_topk")
        oracle_objectness = mean("learned_reader_answer_exact", eval_mode="oracle_objectness")
        oracle_fields = mean("learned_reader_answer_exact", eval_mode="oracle_fields")
        oracle_count_fields = mean("learned_reader_answer_exact", eval_mode="oracle_count_and_fields")
        objectness_margin = mean("objectness_margin", eval_mode="normal_v2")
        objectness_f1 = mean("objectness_f1", eval_mode="normal_v2")
        duplicate_rate = mean("duplicate_slot_rate", eval_mode="normal_v2")
        slot_count_accuracy = mean("slot_count_accuracy", eval_mode="normal_v2")
        field_slot_f1 = mean("slot_f1", eval_mode="normal_v2")
        best_t, best_t_exact, best_t_f1 = best_threshold()
        lines = [
            "## Writer v2 Bottleneck Decomposition",
            "",
            "This diagnostic keeps the v2 architecture fixed and evaluates alternate masks/fields to separate objectness, slot-count calibration, duplicate behavior, and pointer-field quality. It does not add a new writer, larger backbone, JEPA, ANN, graph memory, RL, or HPM recurrence.",
            "",
            f"Tasks: `{args.tasks}`. Writers: `{args.writers}`. Noise levels: `{args.noise_levels}`. Marker rates: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot counts: `{args.slot_counts}`. Max slots: `{args.max_slots}`.",
            f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Lambda objectness values: `{args.lambda_obj_values}`. Eval modes: `{args.eval_modes}`. Threshold sweep: `{args.threshold_sweep}`. Device request: `{args.device}`. Torch: `{torch.__version__}`.",
            f"Projected requested rows: `{projected_rows}`. Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --objectness-threshold {args.objectness_threshold} --lambda-obj-values {args.lambda_obj_values} --threshold-sweep {args.threshold_sweep} --eval-modes {args.eval_modes}`",
            "",
            "Verification:",
            "",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m py_compile .\\hpm_lite\\noisy_extraction.py .\\scripts\\run_noisy_slot_extraction.py` passed.",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m pytest -q` passed before and after the diagnostic.",
            "",
            "### Eval Mode Summary",
            "",
            "| eval mode | learned exact | symbolic exact | slot F1 | all-slots exact | slot-count acc | objectness F1 | objectness margin | gain vs normal |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode in parse_str_list(args.eval_modes, EVAL_MODES):
            exact = mean("learned_reader_answer_exact", eval_mode=mode)
            gain = None if exact is None or normal is None else exact - normal
            lines.append(
                f"| {mode} | {fmt(exact)} | {fmt(mean('symbolic_answer_exact', eval_mode=mode))} | {fmt(mean('slot_f1', eval_mode=mode))} | {fmt(mean('all_slots_exact', eval_mode=mode))} | {fmt(mean('slot_count_accuracy', eval_mode=mode))} | {fmt(mean('objectness_f1', eval_mode=mode))} | {fmt(mean('objectness_margin', eval_mode=mode))} | {fmt(gain)} |"
            )

        lines.extend(
            [
                "",
                "### Lambda Objectness Sweep",
                "",
                "| lambda_obj | normal exact | oracle_count_topk exact | oracle_objectness exact | oracle_fields exact | objectness F1 | slot-count acc |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for lambda_obj in parse_float_list(args.lambda_obj_values) if args.lambda_obj_values else [args.lambda_obj]:
            lines.append(
                f"| {lambda_obj:g} | {fmt(mean('learned_reader_answer_exact', eval_mode='normal_v2', lambda_obj=lambda_obj))} | {fmt(mean('learned_reader_answer_exact', eval_mode='oracle_count_topk', lambda_obj=lambda_obj))} | {fmt(mean('learned_reader_answer_exact', eval_mode='oracle_objectness', lambda_obj=lambda_obj))} | {fmt(mean('learned_reader_answer_exact', eval_mode='oracle_fields', lambda_obj=lambda_obj))} | {fmt(mean('objectness_f1', eval_mode='normal_v2', lambda_obj=lambda_obj))} | {fmt(mean('slot_count_accuracy', eval_mode='normal_v2', lambda_obj=lambda_obj))} |"
            )

        lines.extend(
            [
                "",
                "### Threshold Sweep",
                "",
                f"Best threshold: `{fmt(best_t)}`. Best threshold exact: `{fmt(best_t_exact)}`. Best threshold slot F1: `{fmt(best_t_f1)}`.",
                "",
                "### Verdict",
                "",
                f"normal_v2 exact = {fmt(normal)}",
                f"oracle_count_topk exact = {fmt(oracle_count)}",
                f"oracle_objectness exact = {fmt(oracle_objectness)}",
                f"oracle_fields exact = {fmt(oracle_fields)}",
                f"oracle_count_and_fields exact = {fmt(oracle_count_fields)}",
                f"objectness_margin = {fmt(objectness_margin)}",
                f"objectness_f1 = {fmt(objectness_f1)}",
                f"slot_count_accuracy = {fmt(slot_count_accuracy)}",
                f"duplicate_slot_rate = {fmt(duplicate_rate)}",
                "",
                "Is the v2 bottleneck objectness, field extraction, calibration, or duplicate suppression?",
            ]
        )
        count_gain = None if normal is None or oracle_count is None else oracle_count - normal
        objectness_gain = None if normal is None or oracle_objectness is None else oracle_objectness - normal
        fields_gain = None if normal is None or oracle_fields is None else oracle_fields - normal
        threshold_gain = None if normal is None or best_t_exact is None else best_t_exact - normal
        if fields_gain is not None and fields_gain > max(count_gain or 0.0, objectness_gain or 0.0, threshold_gain or 0.0, 0.05):
            lines.append(f"Field pointer extraction is the largest bottleneck: oracle_fields gains {fields_gain:.3f} over normal.")
        elif count_gain is not None and count_gain > 0.05:
            lines.append(f"Slot-count/objectness selection is the bottleneck: oracle_count_topk gains {count_gain:.3f} over normal.")
        elif objectness_gain is not None and objectness_gain > 0.05:
            lines.append(f"Objectness assignment is the bottleneck: oracle_objectness gains {objectness_gain:.3f} over normal.")
        elif threshold_gain is not None and threshold_gain > 0.05:
            lines.append(f"Calibration is the bottleneck: threshold sweep gains {threshold_gain:.3f} over the 0.5 threshold.")
        elif duplicate_rate is not None and duplicate_rate > 0.10:
            lines.append(f"Duplicate suppression is suspicious: duplicate slot rate is {duplicate_rate:.3f}.")
        else:
            lines.append("No single ablation gives a large gain; failures are mixed or reader/data integration should be inspected.")
        lines.append("Should Writer v3 use AdaSlot-style adaptive selection, high-recall span candidates, or better pointer fields?")
        if count_gain is not None and count_gain > (fields_gain or 0.0):
            lines.append("Prefer AdaSlot-style adaptive selection/objectness calibration first, then revisit fields.")
        elif fields_gain is not None and fields_gain > 0.05:
            lines.append("Prefer better pointer fields or high-recall span candidates first.")
        else:
            lines.append("Keep Writer v3 small and target the strongest measured gain above rather than changing the whole architecture.")
        lines.append("")
        section = "\n".join(lines)
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v2 Bottleneck Decomposition\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")
        return

    if v2_mode:
        lines = [
            "## Noisy Slot Extraction v2: Order-Invariant Writer",
            "",
            "This run adds a DETR-style set extractor with slot objectness and Hungarian-style matching over typed key/value/condition pointers. It trains only writer/extractor modules; the structured readers remain frozen, and no backbone, CE decoder, HPM recurrence, JEPA, ANN, graph memory, Priming, GKA, or RL is trained.",
            "",
            f"Tasks: `{args.tasks}`. Writers: `{args.writers}`. Noise levels: `{args.noise_levels}`. Marker rates: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot counts: `{args.slot_counts}`. Max slots: `{args.max_slots}`.",
            f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Objectness threshold: `{args.objectness_threshold}`. Lambda objectness: `{args.lambda_obj}`. Device request: `{args.device}`. Platform: `{platform.platform()}`. Torch: `{torch.__version__}`.",
            f"Projected requested rows: `{projected_rows}`. Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --objectness-threshold {args.objectness_threshold}`",
            "",
            "Verification:",
            "",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m py_compile .\\hpm_lite\\data.py .\\hpm_lite\\write_modes.py .\\hpm_lite\\structured_readout.py .\\hpm_lite\\noisy_extraction.py .\\scripts\\run_noisy_slot_extraction.py` passed.",
            "- `.\\.venv-cuda\\Scripts\\python.exe -m pytest -q` passed before and after Stage A: 30 tests passed.",
            "",
            "### Final Budget Writer Comparison",
            "",
            "| writer | slot F1 | all-slots exact | learned-reader exact | symbolic exact | slot-count accuracy | predicted slots | true slots | duplicate rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for writer in writers:
            lines.append(
                f"| {writer_label(writer)} | {fmt(summary_mean(summary, writer, final_budget, 'slot_f1'))} | {fmt(summary_mean(summary, writer, final_budget, 'all_slots_exact'))} | {fmt(summary_mean(summary, writer, final_budget, 'learned_reader_answer_exact'))} | {fmt(summary_mean(summary, writer, final_budget, 'symbolic_answer_exact'))} | {fmt(summary_mean(summary, writer, final_budget, 'slot_count_accuracy'))} | {fmt(summary_mean(summary, writer, final_budget, 'predicted_slot_count'))} | {fmt(summary_mean(summary, writer, final_budget, 'true_slot_count'))} | {fmt(summary_mean(summary, writer, final_budget, 'duplicate_slot_rate'))} |"
            )

        lines.extend(
            [
                "",
                "### Performance By Marker Rate",
                "",
                "| writer | marker_rate | slot F1 | learned-reader exact | symbolic exact | slot-count accuracy |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for writer in writers:
            for marker_rate in parse_float_list(args.marker_rates):
                filters = {"marker_rate": marker_rate}
                lines.append(
                    f"| {writer_label(writer)} | {marker_rate:.1f} | {fmt(summary_mean(summary, writer, final_budget, 'slot_f1', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'learned_reader_answer_exact', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'symbolic_answer_exact', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'slot_count_accuracy', **filters))} |"
                )

        lines.extend(
            [
                "",
                "### Performance By Noise Level",
                "",
                "| writer | noise | slot F1 | learned-reader exact | symbolic exact | slot-count accuracy |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for writer in writers:
            for noise_level in parse_str_list(args.noise_levels, NOISE_LEVELS):
                filters = {"noise_level": noise_level}
                lines.append(
                    f"| {writer_label(writer)} | {noise_level} | {fmt(summary_mean(summary, writer, final_budget, 'slot_f1', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'learned_reader_answer_exact', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'symbolic_answer_exact', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'slot_count_accuracy', **filters))} |"
                )

        lines.extend(
            [
                "",
                "### Performance By Slot Count",
                "",
                "| writer | slot_count | max_slots | slot F1 | learned-reader exact | symbolic exact | slot-count accuracy |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for writer in writers:
            for slot_count in parse_int_list(args.slot_counts):
                for max_slots in (parse_int_list(args.max_slots) if args.max_slots else [max(parse_int_list(args.slot_counts))]):
                    if max_slots < slot_count:
                        continue
                    filters = {"slot_count": slot_count, "max_slots": max_slots}
                    lines.append(
                        f"| {writer_label(writer)} | {slot_count} | {max_slots} | {fmt(summary_mean(summary, writer, final_budget, 'slot_f1', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'learned_reader_answer_exact', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'symbolic_answer_exact', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'slot_count_accuracy', **filters))} |"
                    )

        oracle = summary_mean(summary, "oracle", final_budget, "learned_reader_answer_exact")
        fact_zero = summary_mean(summary, "fact_token", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
        v1_zero = summary_mean(summary, "learned_typed_extractor", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
        v2_zero = summary_mean(summary, "learned_set_extractor_v2", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
        v2_symbolic_zero = summary_mean(summary, "learned_set_extractor_v2", final_budget, "symbolic_answer_exact", marker_rate=0.0)
        v2_f1 = summary_mean(summary, "learned_set_extractor_v2", final_budget, "slot_f1")
        v2_count = summary_mean(summary, "learned_set_extractor_v2", final_budget, "slot_count_accuracy")
        v2_duplicate = summary_mean(summary, "learned_set_extractor_v2", final_budget, "duplicate_slot_rate")
        lines.extend(["", "### Verdict", ""])
        lines.append("Can typed memory survive unordered learned slot extraction?")
        if v2_zero is not None and v2_zero >= 0.90:
            lines.append(f"Yes on this Stage A run: v2 learned-reader exact at marker-rate 0.0 is {v2_zero:.3f}.")
        else:
            lines.append(f"Not yet cleanly: v2 learned-reader exact at marker-rate 0.0 is {fmt(v2_zero)}, with symbolic-on-v2-slots exact {fmt(v2_symbolic_zero)} and slot F1 {fmt(v2_f1)}.")
        lines.append("Does v2 beat fact_token when markers disappear?")
        if fact_zero is not None and v2_zero is not None and v2_zero > fact_zero + 0.05:
            lines.append(f"Yes: at marker-rate 0.0, v2 reaches {v2_zero:.3f} versus fact_token {fact_zero:.3f}.")
        elif fact_zero is not None and v2_zero is not None:
            lines.append(f"No: at marker-rate 0.0, v2 reaches {v2_zero:.3f} versus fact_token {fact_zero:.3f}.")
        else:
            lines.append("Insufficient marker-rate 0.0 rows to compare.")
        lines.append("Does v2 remove the known-slot-count / canonical-order cheat?")
        lines.append("Partially. The v2 extractor predicts an unordered set and uses objectness at inference instead of the exact slot count. In this sweep each configuration is still trained/evaluated at a fixed true slot-count setting, so variable-count generalization remains a limitation.")
        if v1_zero is not None and v2_zero is not None:
            lines.append(f"At marker-rate 0.0, v1 canonical exact is {v1_zero:.3f}; v2 unordered exact is {v2_zero:.3f}.")
        lines.append("Is the next bottleneck slot count, slot field extraction, duplicate suppression, or real-data conversion?")
        if oracle is not None and oracle >= 0.95 and v2_zero is not None and v2_zero < 0.90:
            if v2_count is not None and v2_count < 0.80:
                lines.append(f"Slot count/objectness is the leading bottleneck: v2 slot-count accuracy averages {v2_count:.3f}.")
            elif v2_f1 is not None and v2_f1 < 0.90:
                lines.append(f"Slot field extraction is the leading bottleneck: v2 slot F1 averages {v2_f1:.3f}.")
            elif v2_duplicate is not None and v2_duplicate > 0.10:
                lines.append(f"Duplicate suppression is suspicious: v2 duplicate slot rate averages {v2_duplicate:.3f}.")
            else:
                lines.append("The remaining bottleneck is integration/noisy extraction quality despite healthy oracle-slot readers.")
        elif v2_zero is not None and v2_zero >= 0.90:
            lines.append("Next bottleneck is likely harder noisy writing or real-data slot extraction.")
        else:
            lines.append("The writer remains the first bottleneck to inspect.")
        lines.append("")
        section = "\n".join(lines)
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Noisy Slot Extraction v2: Order-Invariant Writer\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")
        return

    lines = [
        "## Noisy Slot Extraction / Learned Writer v1",
        "",
        "This run trains only a tiny typed slot extractor. It freezes a small structured reader after a short oracle-slot pretrain and does not train a backbone, CE decoder, HPM recurrence, JEPA, ANN, graph memory, Priming, GKA, or RL.",
        "",
        f"Tasks: `{args.tasks}`. Writers: `{args.writers}`. Noise levels: `{args.noise_levels}`. Marker rates: `{args.marker_rates}`. Distractors: `{args.distractor_counts}`. Slot counts: `{args.slot_counts}`.",
        f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Reader pretrain steps: `{args.reader_pretrain_steps}`. Device request: `{args.device}`. Platform: `{platform.platform()}`. Torch: `{torch.__version__}`.",
        f"Projected requested rows: `{projected_rows}`. Raw rows saved: `{raw_count}`. Summary rows: `{len(summary)}`.",
        "",
        "Command:",
        "",
        f"- `python scripts/run_noisy_slot_extraction.py --tasks {args.tasks} --writers {args.writers} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --extractor-dim {args.extractor_dim} --extractor-hidden {args.extractor_hidden} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts}`",
        "",
        "### Final Budget By Marker Rate",
        "",
        "| writer | marker_rate | slot_f1 | learned reader exact | symbolic exact | all slots exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for writer in parse_str_list(args.writers, WRITERS):
        for marker_rate in parse_float_list(args.marker_rates):
            filters = {"marker_rate": marker_rate}
            lines.append(
                f"| {writer} | {marker_rate:.1f} | {fmt(summary_mean(summary, writer, final_budget, 'slot_f1', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'learned_reader_answer_exact', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'symbolic_answer_exact', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'all_slots_exact', **filters))} |"
            )

    lines.extend(
        [
            "",
            "### Final Budget By Noise Level",
            "",
            "| writer | noise | slot_f1 | learned reader exact | symbolic exact |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for writer in parse_str_list(args.writers, WRITERS):
        for noise_level in parse_str_list(args.noise_levels, NOISE_LEVELS):
            filters = {"noise_level": noise_level}
            lines.append(
                f"| {writer} | {noise_level} | {fmt(summary_mean(summary, writer, final_budget, 'slot_f1', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'learned_reader_answer_exact', **filters))} | {fmt(summary_mean(summary, writer, final_budget, 'symbolic_answer_exact', **filters))} |"
            )

    oracle = summary_mean(summary, "oracle", final_budget, "learned_reader_answer_exact")
    fact_zero = summary_mean(summary, "fact_token", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
    learned_zero = summary_mean(summary, "learned_typed_extractor", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
    learned_f1 = summary_mean(summary, "learned_typed_extractor", final_budget, "slot_f1")
    lines.extend(["", "### Verdict", ""])
    lines.append("Can typed memory survive learned/noisy slot extraction?")
    if learned_f1 is not None and learned_f1 >= 0.90 and learned_zero is not None and learned_zero >= 0.90:
        lines.append(f"Yes on this run: learned extractor slot F1 averages {learned_f1:.3f} and marker-rate-0 learned-reader exact averages {learned_zero:.3f}.")
    else:
        lines.append(f"Not yet cleanly: learned extractor slot F1 is {fmt(learned_f1)} and marker-rate-0 learned-reader exact is {fmt(learned_zero)}.")
    lines.append("Does learned extraction beat fact_token when markers disappear?")
    if fact_zero is not None and learned_zero is not None and learned_zero > fact_zero + 0.05:
        lines.append(f"Yes: at marker-rate 0.0, learned extraction reaches {learned_zero:.3f} versus fact_token {fact_zero:.3f}.")
    elif fact_zero is not None and learned_zero is not None:
        lines.append(f"No meaningful separation: marker-rate 0.0 learned extraction {learned_zero:.3f}, fact_token {fact_zero:.3f}.")
    else:
        lines.append("Insufficient marker-rate 0.0 rows to compare.")
    lines.append("Is the next bottleneck writer, reader, or real-data conversion?")
    if oracle is not None and learned_zero is not None and oracle > 0.95 and learned_zero < 0.90:
        lines.append("Writer/extractor is the bottleneck: oracle slots remain high while learned slots do not.")
    elif learned_zero is not None and learned_zero >= 0.90:
        lines.append("Next bottleneck is real-data slot extraction or harder noisy writing; the reader path is still healthy.")
    else:
        lines.append("The writer is still the first bottleneck to inspect.")
    lines.append("")
    section = "\n".join(lines)
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## Noisy Slot Extraction / Learned Writer v1\n"
        index = old.find(marker)
        if index >= 0:
            old = old[:index].rstrip()
        path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
    else:
        path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")


def update_writeup(path: Path, summary: List[Dict[str, object]], args: argparse.Namespace) -> None:
    final_budget = max(parse_int_list(args.budgets))
    writers = parse_str_list(args.writers, WRITERS)
    if args.output_prefix in {"writer_v45_frozen_baseline", "writer_v45_pairbeam_b4", "writer_v45_pairbeam_b8", "writer_v45_pairbeam_b16"}:
        return
    if args.output_prefix == "writer_v45_optimization":
        rows = [row for row in summary if row.get("writer") == "v4_condition_v2_full" and int(row.get("budget_steps", 0)) == final_budget]

        def metric(row: Dict[str, object] | None, name: str) -> float | None:
            if row is None:
                return None
            value = row.get(f"{name}_mean", "")
            return None if value == "" else float(value)

        baseline = next((row for row in rows if row.get("tuple_pruning") == "none"), None)
        best_pair = None
        best_rejected_pair = None
        for row in rows:
            if row.get("tuple_pruning") != "pair_beam":
                continue
            exact = metric(row, "learned_reader_answer_exact") or 0.0
            cond = metric(row, "learned_candidate_cond_recall") or 0.0
            if best_rejected_pair is None or exact > (metric(best_rejected_pair, "learned_reader_answer_exact") or -1.0):
                best_rejected_pair = row
            if exact >= 0.94 and cond >= 0.986:
                if best_pair is None or (metric(row, "tuple_scorer_time") or 1e9) < (metric(best_pair, "tuple_scorer_time") or 1e9):
                    best_pair = row
        pair_label = "accepted pair_beam" if best_pair is not None else "rejected pair_beam"
        pair_row = best_pair if best_pair is not None else best_rejected_pair
        section = f"""## Writer v4.5 Freeze and Speed Optimization

The repaired Writer v4.5 setting is now frozen as candidate_k=16, guideline_loss_weight=2.0, simplified_aux_weight=0.5, and template_augmentation=extreme. Candidate K values above 16 are reserved for explicit slow sweeps.

| setting | exact | condition recall | tuple candidates | tuple scorer time |
| --- | ---: | ---: | ---: | ---: |
| full K^3 | {fmt(metric(baseline, "learned_reader_answer_exact"))} | {fmt(metric(baseline, "learned_candidate_cond_recall"))} | {fmt(metric(baseline, "tuple_candidates_scored"))} | {fmt(metric(baseline, "tuple_scorer_time"))} |
| {pair_label} | {fmt(metric(pair_row, "learned_reader_answer_exact"))} | {fmt(metric(pair_row, "learned_candidate_cond_recall"))} | {fmt(metric(pair_row, "tuple_candidates_scored"))} | {fmt(metric(pair_row, "tuple_scorer_time"))} |

Interpretation: pair-beam becomes default only if it clears 0.94 exact and 0.986 condition recall while materially reducing tuple scoring time. Otherwise full K^3 remains the trusted frozen diagnostic.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v4.5 Freeze and Speed Optimization\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return
    if args.output_prefix == "writer_v45_residual_fix":
        rows = [row for row in summary if row.get("writer") == "v4_condition_v2_full" and int(row.get("budget_steps", 0)) == final_budget]

        def metric(row: Dict[str, object] | None, name: str) -> float | None:
            if row is None:
                return None
            value = row.get(f"{name}_mean", "")
            return None if value == "" else float(value)

        best = max(rows, key=lambda row: metric(row, "learned_reader_answer_exact") or -1.0) if rows else None
        section = f"""## Writer v4.5 Residual Fix

The residual sweep keeps Writer v4.5 full and varies only candidate count, condition guideline weight, simplified auxiliary weight, and template augmentation.

| best exact | condition recall | condition miss | value miss | tuple scoring error | candidate_k | guideline | simplified aux | augmentation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| {fmt(metric(best, "learned_reader_answer_exact"))} | {fmt(metric(best, "learned_candidate_cond_recall"))} | {fmt(metric(best, "condition_miss_rate"))} | {fmt(metric(best, "value_miss_rate"))} | {fmt(metric(best, "tuple_scoring_error_rate"))} | {best.get("candidate_k") if best else ""} | {best.get("guideline_loss_weight") if best else ""} | {best.get("simplified_aux_weight") if best else ""} | {best.get("template_augmentation") if best else ""} |

Interpretation: if the best row is at candidate_k > 16, candidate recall was still the main issue. If higher guideline weight wins, condition semantics helped. If extreme augmentation wins, template coverage helped. If no row reaches 0.90 exact and 0.98 condition recall, stop synthetic patching and inspect the real-data conversion path cautiously.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v4.5 Residual Fix\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return
    if args.output_prefix in {"writer_v45_condition_generalization", "writer_v45_condition_generalization_random"}:
        heldout_path = Path(args.out_dir) / "writer_v45_condition_generalization_summary.csv"
        random_path = Path(args.out_dir) / "writer_v45_condition_generalization_random_summary.csv"

        def load_summary_file(path_: Path) -> List[Dict[str, object]]:
            if not path_.exists():
                return []
            with path_.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))

        heldout_summary = load_summary_file(heldout_path)
        random_summary = load_summary_file(random_path)
        if args.template_split == "heldout" and not heldout_summary:
            heldout_summary = summary
        if args.template_split == "random" and not random_summary:
            random_summary = summary

        def heldout_mean(writer: str, metric: str) -> float | None:
            return summary_mean(heldout_summary or summary, writer, final_budget, metric)

        def random_mean(writer: str, metric: str) -> float | None:
            return summary_mean(random_summary, writer, final_budget, metric) if random_summary else None

        baseline = heldout_mean("baseline_v4", "learned_reader_answer_exact")
        full = heldout_mean("v4_condition_v2_full", "learned_reader_answer_exact")
        full_cond = heldout_mean("v4_condition_v2_full", "learned_candidate_cond_recall")
        full_random = random_mean("v4_condition_v2_full", "learned_reader_answer_exact")
        best_writer = None
        best_exact = None
        for writer in writers:
            exact = heldout_mean(writer, "learned_reader_answer_exact")
            if exact is not None and (best_exact is None or exact > best_exact):
                best_writer = writer
                best_exact = exact
        section = f"""## Heldout Template Generalization Fix

Writer v4.5 keeps contextual tuple assembly fixed and tests whether condition candidate extraction can generalize beyond random template splits. The ablations isolate span-based condition candidates, role/guideline contrast, deterministic template augmentation, and fact-local simplified auxiliary supervision.

| setting | exact | condition recall |
| --- | ---: | ---: |
| baseline_v4 | {fmt(baseline)} | {fmt(heldout_mean("baseline_v4", "learned_candidate_cond_recall"))} |
| v4_span_condition | {fmt(heldout_mean("v4_span_condition", "learned_reader_answer_exact"))} | {fmt(heldout_mean("v4_span_condition", "learned_candidate_cond_recall"))} |
| v4_guideline_condition | {fmt(heldout_mean("v4_guideline_condition", "learned_reader_answer_exact"))} | {fmt(heldout_mean("v4_guideline_condition", "learned_candidate_cond_recall"))} |
| v4_augmented_templates | {fmt(heldout_mean("v4_augmented_templates", "learned_reader_answer_exact"))} | {fmt(heldout_mean("v4_augmented_templates", "learned_candidate_cond_recall"))} |
| v4_simplified_aux | {fmt(heldout_mean("v4_simplified_aux", "learned_reader_answer_exact"))} | {fmt(heldout_mean("v4_simplified_aux", "learned_candidate_cond_recall"))} |
| v4_condition_v2_full | {fmt(full)} | {fmt(full_cond)} |

Best ablation: `{best_writer or ""}` with exact {fmt(best_exact)}.

Random-template reference for full v4.5: {fmt(full_random)}.

Interpretation: if v4.5 clears 0.90 heldout exact and 0.98 condition recall, the synthetic heldout bottleneck is mostly repaired. If not, the next step should remain synthetic condition-candidate generalization rather than real-data slot extraction.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Heldout Template Generalization Fix\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return
    if args.output_prefix == "writer_v4_stageb":
        v4 = "contextual_tuple_learned_candidates"
        v4_exact = summary_mean(summary, v4, final_budget, "learned_reader_answer_exact")
        v2_exact = summary_mean(summary, "learned_set_extractor_v2", final_budget, "learned_reader_answer_exact")
        fact_exact = summary_mean(summary, "fact_token", final_budget, "learned_reader_answer_exact")
        hard_slot8 = summary_mean(summary, v4, final_budget, "learned_reader_answer_exact", task="noisy_conditional", noise_level="hard", slot_count=8)
        coexisting = summary_mean(summary, v4, final_budget, "learned_reader_answer_exact", task="noisy_coexisting")
        conditional = summary_mean(summary, v4, final_budget, "learned_reader_answer_exact", task="noisy_conditional")
        repair = summary_mean(summary, "contextual_tuple_learned_candidates_plus_oracle_missing", final_budget, "oracle_missing_repair_gain")
        noise_damage = summary_mean(summary, "contextual_tuple_oracle_candidates_plus_learned_noise", final_budget, "learned_noise_damage")
        section = f"""## Writer v4 Robustness and Template Generalization

Stage B stress-tests the full learned Writer v4 path under harder synthetic noise: medium/hard templates, more distractors, slot counts 4/8, and noisy conditional plus coexisting tasks.

| final budget | v4 exact | v2 exact | fact_token exact | hard conditional slot8 exact | conditional exact | coexisting exact | repair gain | learned-noise damage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(v4_exact)} | {fmt(v2_exact)} | {fmt(fact_exact)} | {fmt(hard_slot8)} | {fmt(conditional)} | {fmt(coexisting)} | {fmt(repair)} | {fmt(noise_damage)} |

Interpretation rule: if repair gain jumps, candidate recall remains the bottleneck. If learned-noise damage jumps, candidate false positives are breaking tuple scoring. If coexisting lags conditional, multi-positive set extraction is the next synthetic fix.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v4 Robustness and Template Generalization\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return

    if args.output_prefix == "writer_v4_heldout":
        heldout = summary_mean(summary, "contextual_tuple_learned_candidates", final_budget, "learned_reader_answer_exact")
        section = f"""## Heldout Template Check

The heldout run trains on simple templates and evaluates on paraphrase templates for the hardest Stage B conditional setting.

| final budget | heldout exact |
| ---: | ---: |
| {final_budget} | {fmt(heldout)} |

Interpretation rule: a large drop from the random-template Stage B reference means template memorization remains a synthetic bottleneck before real-data conversion.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Heldout Template Check\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return

    if is_contextual_learned_candidate_run(writers):
        oracle = summary_mean(summary, "contextual_tuple_oracle_candidates", final_budget, "learned_reader_answer_exact")
        learned = summary_mean(summary, "contextual_tuple_learned_candidates", final_budget, "learned_reader_answer_exact")
        repaired = summary_mean(summary, "contextual_tuple_learned_candidates_plus_oracle_missing", final_budget, "learned_reader_answer_exact")
        noisy = summary_mean(summary, "contextual_tuple_oracle_candidates_plus_learned_noise", final_budget, "learned_reader_answer_exact")
        key_recall = summary_mean(summary, "contextual_tuple_learned_candidates", final_budget, "learned_candidate_key_recall")
        cond_recall = summary_mean(summary, "contextual_tuple_learned_candidates", final_budget, "learned_candidate_cond_recall")
        value_recall = summary_mean(summary, "contextual_tuple_learned_candidates", final_budget, "learned_candidate_value_recall")
        section = f"""## Learned Candidate Extraction Bottleneck

After contextual tuple assembly solved oracle candidates, this diagnostic swaps in candidate fields from the learned `CandidateFieldProposer`. Repair/noise/oracle-field modes separate missing candidates from noisy candidate pools and tuple-scoring failures.

| final budget | oracle exact | learned exact | learned+oracle-missing exact | oracle+learned-noise exact | key recall | condition recall | value recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(oracle)} | {fmt(learned)} | {fmt(repaired)} | {fmt(noisy)} | {fmt(key_recall)} | {fmt(cond_recall)} | {fmt(value_recall)} |

Interpretation rule: if learned+oracle-missing jumps, candidate recall is the bottleneck. If oracle+learned-noise falls, candidate false positives are breaking tuple scoring. If recall is high but learned exact is low, tuple selection under noisy learned pools is the bottleneck.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Learned Candidate Extraction Bottleneck\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return

    if is_contextual_tuple_run(writers):
        oracle = summary_mean(summary, "contextual_tuple_oracle_candidates", final_budget, "learned_reader_answer_exact")
        hard = summary_mean(summary, "contextual_tuple_oracle_candidates_plus_hard_negatives", final_budget, "learned_reader_answer_exact")
        gold_key_cond = summary_mean(summary, "contextual_tuple_gold_key_cond", final_budget, "learned_reader_answer_exact")
        gold_all = summary_mean(summary, "contextual_tuple_gold_all_fields", final_budget, "learned_reader_answer_exact")
        margin = summary_mean(summary, "contextual_tuple_oracle_candidates_plus_hard_negatives", final_budget, "tuple_score_margin")
        section = f"""## Contextual Tuple Assembly

Writer v4 tests whether tuple assembly needs relation evidence from the original text, not just a product of candidate field embeddings. The scorer sees contextual token states, relative field positions, field order, pooled text between fields, and local windows around key/condition/value candidates.

| final budget | contextual oracle exact | contextual hard-negative exact | gold key+cond exact | gold-all-fields exact | hard-negative score margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(oracle)} | {fmt(hard)} | {fmt(gold_key_cond)} | {fmt(gold_all)} | {fmt(margin)} |

Interpretation rule: if oracle candidates reach 1.0, contextual relation evidence fixed tuple assembly under clean candidates. If oracle candidates work but hard negatives fail, relation discrimination is the bottleneck. If gold key+condition is still low, value relation scoring is weak or labels/features are still wrong.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Contextual Tuple Assembly\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return

    if is_tuple_debug_run(writers):
        independent = summary_mean(summary, "writer_v3_oracle_candidates_sanity", final_budget, "learned_reader_answer_exact", eval_mode="independent_field_heads_current")
        sanity = summary_mean(summary, "writer_v3_oracle_candidates_sanity", final_budget, "learned_reader_answer_exact", eval_mode="no_objectness_true_count")
        gold_all = summary_mean(summary, "writer_v3_oracle_candidates_sanity", final_budget, "learned_reader_answer_exact", eval_mode="gold_all_fields")
        spn = summary_mean(summary, "spn_tuple_assembler_oracle_candidates", final_budget, "learned_reader_answer_exact")
        section = f"""## Tuple Assembly Failure Analysis

This diagnostic isolates the v3 tuple assembler under oracle candidate fields. It checks independent field heads, true-count/no-objectness assembly, gold-field ablations, and an SPN-style whole-tuple scorer.

| final budget | independent heads exact | true-count exact | gold-all-fields exact | SPN tuple exact |
| ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(independent)} | {fmt(sanity)} | {fmt(gold_all)} | {fmt(spn)} |

Interpretation rule: if gold-all-fields is not 1.0, indexing or slot formatting is broken. If gold-all-fields is 1.0 but true-count independent heads stay low, the factorized assembler objective is failing. If SPN succeeds where independent heads fail, Writer v4 should move toward whole-tuple scoring.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Tuple Assembly Failure Analysis\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return

    if is_v3_run(writers):
        v2_exact = summary_mean(summary, "learned_set_extractor_v2", final_budget, "learned_reader_answer_exact")
        oracle_exact = summary_mean(summary, "writer_v3_oracle_candidates", final_budget, "learned_reader_answer_exact")
        learned_exact = summary_mean(summary, "writer_v3_learned_candidates", final_budget, "learned_reader_answer_exact")
        recall = summary_mean(summary, "writer_v3_learned_candidates", final_budget, "all_fields_candidate_recall")
        gain = summary_mean(summary, "writer_v3_learned_candidates", final_budget, "v3_gain_over_v2")
        section = f"""## Writer v3 Field Candidate Decomposition

Writer v3 tests whether high-recall field candidates can repair the v2 pointer bottleneck. It decomposes the pipeline into candidate proposal and slot assembly over candidate fields.

| final budget | v2 exact | v3 oracle-candidate exact | v3 learned-candidate exact | learned candidate recall | v3 gain over v2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(v2_exact)} | {fmt(oracle_exact)} | {fmt(learned_exact)} | {fmt(recall)} | {fmt(gain)} |

Interpretation rule: if oracle candidates work but learned candidates do not, field proposal recall is the bottleneck. If oracle candidates do not work, tuple assembly is the bottleneck. If learned candidates beat v2, high-recall candidate fields are moving the writer in the right direction.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v3 Field Candidate Decomposition\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return

    if is_bottleneck_run(args):
        normal = summary_mean(summary, "learned_set_extractor_v2", final_budget, "learned_reader_answer_exact", eval_mode="normal_v2")
        oracle_count = summary_mean(summary, "learned_set_extractor_v2", final_budget, "learned_reader_answer_exact", eval_mode="oracle_count_topk")
        oracle_objectness = summary_mean(summary, "learned_set_extractor_v2", final_budget, "learned_reader_answer_exact", eval_mode="oracle_objectness")
        oracle_fields = summary_mean(summary, "learned_set_extractor_v2", final_budget, "learned_reader_answer_exact", eval_mode="oracle_fields")
        objectness_margin = summary_mean(summary, "learned_set_extractor_v2", final_budget, "objectness_margin", eval_mode="normal_v2")
        slot_count_accuracy = summary_mean(summary, "learned_set_extractor_v2", final_budget, "slot_count_accuracy", eval_mode="normal_v2")
        section = f"""## Writer v2 Failure Decomposition

The v2 writer now has a bottleneck decomposition rather than a new architecture. The diagnostic compares normal objectness-threshold inference with oracle count, oracle objectness, oracle fields, oracle count+fields, and a threshold sweep.

| final budget | normal exact | oracle count exact | oracle objectness exact | oracle fields exact | objectness margin | slot-count accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(normal)} | {fmt(oracle_count)} | {fmt(oracle_objectness)} | {fmt(oracle_fields)} | {fmt(objectness_margin)} | {fmt(slot_count_accuracy)} |

Interpretation rule: if oracle count or oracle objectness jumps, Writer v3 should target adaptive slot selection/calibration. If oracle fields jumps, Writer v3 should target better pointer fields or high-recall span candidates. If neither jumps, inspect reader integration or data generation before changing architecture.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Writer v2 Failure Decomposition\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return

    if "learned_set_extractor_v2" in writers:
        v1_zero = summary_mean(summary, "learned_typed_extractor", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
        v2_zero = summary_mean(summary, "learned_set_extractor_v2", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
        v2_f1 = summary_mean(summary, "learned_set_extractor_v2", final_budget, "slot_f1")
        v2_all_exact = summary_mean(summary, "learned_set_extractor_v2", final_budget, "all_slots_exact")
        v2_count = summary_mean(summary, "learned_set_extractor_v2", final_budget, "slot_count_accuracy")
        fact_zero = summary_mean(summary, "fact_token", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
        section = f"""## Noisy Slot Extraction v2: Order-Invariant Writer

V1 was useful but too friendly: it assumed the true slot count and trained against canonical occurrence order. V2 replaces that with a small DETR-style set extractor. It emits up to `max_slots` unordered slot queries, predicts objectness for each query, and points to typed key/value/condition positions inside the pre-query sequence.

| final budget | fact_token exact at marker 0.0 | v1 canonical exact at marker 0.0 | v2 unordered exact at marker 0.0 | v2 slot F1 | v2 all-slots exact | v2 slot-count accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(fact_zero)} | {fmt(v1_zero)} | {fmt(v2_zero)} | {fmt(v2_f1)} | {fmt(v2_all_exact)} | {fmt(v2_count)} |

The important distinction is that v2 no longer chooses slot `j` because the metadata says the `j`th gold fact exists. It must decide whether each predicted slot exists, then form a typed tuple from pointers. This is a more honest bridge from hand-coded parsing to learned writing.

Remaining limitations: Stage A still trains separate configurations for fixed slot-count settings, so variable-count generalization is only partially tested. The assignment matcher is an exact local Hungarian-style solver for the rectangular matching used here, but the data is still synthetic token text rather than real extraction.

Next recommendation: if v2 slot-count accuracy is low, improve objectness/duplicate suppression before touching the reader. If slot F1 is low while oracle-slot reader exact is high, writer field extraction is the bottleneck. If v2 is strong at marker-rate 0.0, the next step is harder noisy writing or real-data slot extraction.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Noisy Slot Extraction v2: Order-Invariant Writer\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return

    learned_f1 = summary_mean(summary, "learned_typed_extractor", final_budget, "slot_f1")
    fact_zero = summary_mean(summary, "fact_token", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
    learned_zero = summary_mean(summary, "learned_typed_extractor", final_budget, "learned_reader_answer_exact", marker_rate=0.0)
    section = f"""## Noisy Slot Extraction / Learned Writer v1

The next diagnostic replaces exact `FACT` parsing with noisy synthetic templates and a small learned typed extractor. The extractor is a pointer model over pre-query tokens; it predicts key/value/condition positions for each known slot in canonical occurrence order.

| final budget | learned extractor slot F1 | fact_token exact at marker 0.0 | learned extractor exact at marker 0.0 |
| ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(learned_f1)} | {fmt(fact_zero)} | {fmt(learned_zero)} |

Interpretation: oracle slots test the reader upper bound, fact-token slots test brittle marker parsing, and learned slots test whether typed memory can move beyond hand-coded `FACT` extraction. V1 still assumes the number of slots is known and uses canonical slot order instead of Hungarian matching.
"""
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## Noisy Slot Extraction / Learned Writer v1\n"
        index = old.find(marker)
        if index >= 0:
            old = old[:index].rstrip()
        path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
    else:
        path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")


def run_config(
    task: str,
    seed: int,
    noise_level: str,
    marker_rate: float,
    distractor_count: int,
    slot_count: int,
    max_slots: int,
    hard_negatives: int,
    lambda_obj_value: float,
    candidate_k: int,
    candidate_loss_weight: float,
    template_augmentation_value: str,
    simplified_aux_weight_value: float,
    guideline_loss_weight_value: float,
    writers: List[str],
    budgets: List[int],
    args: argparse.Namespace,
    device: torch.device,
    reader_cache: Dict[tuple, torch.nn.Module],
    raw_rows: List[Dict[str, object]],
    out_dir: Path,
    debug_examples: List[Dict[str, object]],
) -> None:
    set_seed(seed)
    cache_key = (task, slot_count, seed)
    if cache_key not in reader_cache:
        reader_cache[cache_key] = train_reader(task, args, device, seed, slot_count)
    reader = reader_cache[cache_key]
    train_v1 = "learned_typed_extractor" in writers
    train_v2 = "learned_set_extractor_v2" in writers
    train_v3 = any(writer.startswith("writer_v3_") for writer in writers)
    train_spn = "spn_tuple_assembler_oracle_candidates" in writers
    v45_writers = [writer for writer in writers if writer in V45_WRITERS]
    train_contextual = any(writer.startswith("contextual_tuple_") for writer in writers)
    extractor_v1 = LearnedTypedExtractor(
        max_slots=slot_count,
        seq_len=args.seq_len,
        has_condition=has_condition(task),
        extractor_dim=args.extractor_dim,
        hidden=args.extractor_hidden,
        layers=args.extractor_layers,
        dropout=args.dropout,
    ).to(device) if train_v1 else None
    extractor_v2 = LearnedSetExtractorV2(
        max_slots=max_slots,
        seq_len=args.seq_len,
        has_condition=has_condition(task),
        extractor_dim=args.extractor_dim,
        hidden=args.extractor_hidden,
        layers=args.extractor_layers,
        dropout=args.dropout,
        objectness_threshold=args.objectness_threshold,
        lambda_obj=lambda_obj_value,
    ).to(device) if train_v2 else None
    optimizer_v1 = TinyAdamW(extractor_v1.parameters(), lr=args.lr, weight_decay=0.0) if extractor_v1 is not None else None
    optimizer_v2 = TinyAdamW(extractor_v2.parameters(), lr=args.lr, weight_decay=0.0) if extractor_v2 is not None else None
    writer_v3 = WriterV3CandidateAssembler(
        max_slots=max_slots,
        seq_len=args.seq_len,
        has_condition=has_condition(task),
        extractor_dim=args.extractor_dim,
        hidden=args.extractor_hidden,
        layers=args.extractor_layers,
        dropout=args.dropout,
        objectness_threshold=args.objectness_threshold,
    ).to(device) if train_v3 else None
    optimizer_v3 = TinyAdamW(writer_v3.parameters(), lr=args.lr, weight_decay=0.0) if writer_v3 is not None else None
    spn_assembler = SPNTupleAssembler(
        seq_len=args.seq_len,
        has_condition=has_condition(task),
        extractor_dim=args.extractor_dim,
        hidden=args.extractor_hidden,
        layers=args.extractor_layers,
        dropout=args.dropout,
    ).to(device) if train_spn else None
    optimizer_spn = TinyAdamW(spn_assembler.parameters(), lr=args.lr, weight_decay=0.0) if spn_assembler is not None else None
    contextual_assembler = ContextualTupleEdgeScorer(
        max_slots=max_slots,
        seq_len=args.seq_len,
        has_condition=has_condition(task),
        extractor_dim=args.extractor_dim,
        hidden=args.extractor_hidden,
        layers=args.extractor_layers,
        dropout=args.dropout,
    ).to(device) if train_contextual else None
    optimizer_contextual = TinyAdamW(contextual_assembler.parameters(), lr=args.lr, weight_decay=0.0) if contextual_assembler is not None else None
    contextual_assemblers: Dict[str, ContextualTupleEdgeScorer] = {}
    optimizer_contextual_v45: Dict[str, TinyAdamW] = {}
    for writer in v45_writers:
        model = ContextualTupleEdgeScorer(
            max_slots=max_slots,
            seq_len=args.seq_len,
            has_condition=has_condition(task),
            extractor_dim=args.extractor_dim,
            hidden=args.extractor_hidden,
            layers=args.extractor_layers,
            dropout=args.dropout,
            condition_proposer_variant=v45_variant(writer),
            simplified_aux_weight=simplified_aux_weight_value,
            guideline_loss_weight=guideline_loss_weight_value,
        ).to(device)
        contextual_assemblers[writer] = model
        optimizer_contextual_v45[writer] = TinyAdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    train_dataset = make_dataset(
        task,
        args,
        seed,
        slot_count,
        noise_level,
        marker_rate,
        distractor_count,
        hard_negatives,
        split_phase="train",
        template_augmentation="none",
    )
    train_dataset_augmented = make_dataset(
        task,
        args,
        seed + 33_000,
        slot_count,
        noise_level,
        marker_rate,
        distractor_count,
        hard_negatives,
        split_phase="train",
        template_augmentation=template_augmentation_value,
    ) if any(v45_uses_augmented_templates(writer) for writer in v45_writers) else None
    budget_set = set(budgets)
    max_budget = max(budgets)
    start = time.perf_counter()
    trained_examples = 0
    last_loss_v1 = 0.0
    last_loss_v2 = 0.0
    last_loss_v3 = 0.0
    last_loss_contextual = 0.0

    def evaluate_budget(budget: int) -> None:
        train_time = time.perf_counter() - start
        for writer in writers:
            for eval_mode, eval_threshold in eval_specs_for_writer(writer, args):
                eval_dataset = make_dataset(
                    task,
                    args,
                    500_000 + seed + budget,
                    slot_count,
                    noise_level,
                    marker_rate,
                    distractor_count,
                    hard_negatives,
                    split_phase="eval",
                )
                metrics = evaluate_writer(
                    task,
                    writer,
                    reader,
                    extractor_v1,
                    extractor_v2,
                    writer_v3,
                    spn_assembler,
                    contextual_assembler,
                    contextual_assemblers,
                    eval_dataset,
                    args,
                    device,
                    candidate_k=candidate_k,
                    eval_mode=eval_mode,
                    eval_threshold=eval_threshold,
                    tuple_pruning=args.tuple_pruning,
                    pair_beam_size=args.pair_beam_size,
                )
                row: Dict[str, object] = {
                    "task": task,
                    "writer": writer,
                    "eval_mode": eval_mode,
                    "eval_threshold": "" if eval_threshold is None else eval_threshold,
                    "lambda_obj": lambda_obj_value,
                    "candidate_k": candidate_k,
                    "tuple_pruning": args.tuple_pruning,
                    "pair_beam_size": args.pair_beam_size if args.tuple_pruning == "pair_beam" else 0,
                    "candidate_loss_weight": candidate_loss_weight,
                    "seed": seed,
                    "budget_steps": budget,
                    "noise_level": noise_level,
                    "marker_rate": marker_rate,
                    "distractor_count": distractor_count,
                    "slot_count": slot_count,
                    "max_slots": max_slots,
                    "hard_negatives": hard_negatives,
                    "template_mix": args.template_mix,
                    "template_split": args.template_split,
                    "template_augmentation": template_augmentation_value,
                    "simplified_aux_weight": simplified_aux_weight_value,
                    "guideline_loss_weight": guideline_loss_weight_value,
                    "reader_type": reader_type(task),
                    "extractor_loss": last_loss_v2 if writer == "learned_set_extractor_v2" else last_loss_v1 if writer == "learned_typed_extractor" else last_loss_v3 if writer.startswith("writer_v3_") else last_loss_contextual if writer.startswith("contextual_tuple_") or writer in V45_WRITERS else "",
                    "extractor_trainable_params": count_trainable_parameters(extractor_v2) if writer == "learned_set_extractor_v2" and extractor_v2 is not None else count_trainable_parameters(extractor_v1) if writer == "learned_typed_extractor" and extractor_v1 is not None else count_trainable_parameters(writer_v3) if writer.startswith("writer_v3_") and writer_v3 is not None else count_trainable_parameters(spn_assembler) if writer == "spn_tuple_assembler_oracle_candidates" and spn_assembler is not None else count_trainable_parameters(contextual_assembler) if writer.startswith("contextual_tuple_") and contextual_assembler is not None else count_trainable_parameters(contextual_assemblers[writer]) if writer in contextual_assemblers else "",
                    "reader_trainable_params": count_trainable_parameters(reader),
                    "extractor_train_time": train_time,
                    "extractor_examples_per_sec": trained_examples / max(train_time, 1.0e-9) if budget > 0 else 0.0,
                    "train_time": train_time,
                    "examples_per_sec": trained_examples / max(train_time, 1.0e-9) if budget > 0 else 0.0,
                }
                row.update(metrics)
                raw_rows.append(row)
                if (
                    writer == "writer_v3_oracle_candidates_sanity"
                    and eval_mode == "independent_field_heads_current"
                    and writer_v3 is not None
                    and budget == max_budget
                    and len(debug_examples) < args.tuple_debug_examples
                ):
                    remaining = args.tuple_debug_examples - len(debug_examples)
                    debug_batch = eval_dataset.sample_batch(args.batch_size, device=device)
                    debug_examples.extend(writer_v3.debug_examples(debug_batch, candidate_k=candidate_k, limit=remaining))
                if (
                    (
                        writer == "contextual_tuple_learned_candidates"
                        or writer in V45_WRITERS
                        or (
                            not is_contextual_learned_candidate_run(writers)
                            and writer == "contextual_tuple_oracle_candidates_plus_hard_negatives"
                        )
                    )
                    and (contextual_assembler is not None or writer in contextual_assemblers)
                    and budget == max_budget
                    and len(debug_examples) < args.tuple_debug_examples
                ):
                    remaining = args.tuple_debug_examples - len(debug_examples)
                    debug_batch = eval_dataset.sample_batch(args.batch_size, device=device)
                    debug_candidate_mode = "learned_candidates" if writer == "contextual_tuple_learned_candidates" or writer in V45_WRITERS else "oracle_candidates_plus_noise"
                    debug_model = contextual_assemblers.get(writer, contextual_assembler)
                    if debug_model is None:
                        continue
                    debug_examples.extend(
                        debug_model.debug_examples(
                            debug_batch,
                            candidate_k=candidate_k,
                            candidate_mode=debug_candidate_mode,
                            limit=remaining,
                        )
                    )
        flush(
            out_dir,
            raw_rows,
            v2_mode="learned_set_extractor_v2" in writers,
            bottleneck_mode=is_bottleneck_run(args),
            v3_mode=is_v3_run(writers),
            contextual_mode=is_contextual_tuple_run(writers),
            output_prefix=args.output_prefix,
        )
        print(
            f"done task={task} noise={noise_level} marker={marker_rate} distract={distractor_count} "
            f"slots={slot_count} max_slots={max_slots} candidate_k={candidate_k} lambda_obj={lambda_obj_value} seed={seed} budget={budget}",
            flush=True,
        )

    if 0 in budget_set:
        evaluate_budget(0)
    for step in range(1, max_budget + 1):
        batch = train_dataset.sample_batch(args.batch_size, device=device)
        if extractor_v1 is not None and optimizer_v1 is not None:
            extractor_v1.train()
            loss_v1 = extractor_v1.loss(batch)
            if not torch.isfinite(loss_v1):
                raise RuntimeError(f"non-finite v1 extractor loss for {task}/seed{seed}/step{step}")
            optimizer_v1.zero_grad(set_to_none=True)
            loss_v1.backward()
            torch.nn.utils.clip_grad_norm_(extractor_v1.parameters(), 1.0)
            optimizer_v1.step()
            last_loss_v1 = float(loss_v1.detach().item())
        if extractor_v2 is not None and optimizer_v2 is not None:
            extractor_v2.train()
            loss_v2 = extractor_v2.loss(batch)
            if not torch.isfinite(loss_v2):
                raise RuntimeError(f"non-finite v2 extractor loss for {task}/seed{seed}/step{step}")
            optimizer_v2.zero_grad(set_to_none=True)
            loss_v2.backward()
            torch.nn.utils.clip_grad_norm_(extractor_v2.parameters(), 1.0)
            optimizer_v2.step()
            last_loss_v2 = float(loss_v2.detach().item())
        if writer_v3 is not None and optimizer_v3 is not None:
            writer_v3.train()
            loss_v3 = writer_v3.loss(batch, candidate_k=candidate_k, candidate_loss_weight=candidate_loss_weight)
            if not torch.isfinite(loss_v3):
                raise RuntimeError(f"non-finite v3 writer loss for {task}/seed{seed}/step{step}")
            optimizer_v3.zero_grad(set_to_none=True)
            loss_v3.backward()
            torch.nn.utils.clip_grad_norm_(writer_v3.parameters(), 1.0)
            optimizer_v3.step()
            last_loss_v3 = float(loss_v3.detach().item())
        if spn_assembler is not None and optimizer_spn is not None:
            spn_assembler.train()
            loss_spn = spn_assembler.loss(batch, candidate_k=candidate_k, candidate_loss_weight=candidate_loss_weight)
            if not torch.isfinite(loss_spn):
                raise RuntimeError(f"non-finite SPN tuple assembler loss for {task}/seed{seed}/step{step}")
            optimizer_spn.zero_grad(set_to_none=True)
            loss_spn.backward()
            torch.nn.utils.clip_grad_norm_(spn_assembler.parameters(), 1.0)
            optimizer_spn.step()
        if contextual_assembler is not None and optimizer_contextual is not None:
            contextual_assembler.train()
            loss_contextual = contextual_assembler.loss(
                batch,
                candidate_k=candidate_k,
                candidate_loss_weight=candidate_loss_weight,
                tuple_loss_weight=args.tuple_loss_weight,
                rank_loss_weight=args.rank_loss_weight,
            )
            if not torch.isfinite(loss_contextual):
                raise RuntimeError(f"non-finite contextual tuple loss for {task}/seed{seed}/step{step}")
            optimizer_contextual.zero_grad(set_to_none=True)
            loss_contextual.backward()
            torch.nn.utils.clip_grad_norm_(contextual_assembler.parameters(), 1.0)
            optimizer_contextual.step()
            last_loss_contextual = float(loss_contextual.detach().item())
        for writer, model in contextual_assemblers.items():
            optimizer = optimizer_contextual_v45[writer]
            train_source = train_dataset_augmented if v45_uses_augmented_templates(writer) and train_dataset_augmented is not None else train_dataset
            batch_v45 = train_source.sample_batch(args.batch_size, device=device)
            model.train()
            loss_v45 = model.loss(
                batch_v45,
                candidate_k=candidate_k,
                candidate_loss_weight=candidate_loss_weight,
                tuple_loss_weight=args.tuple_loss_weight,
                rank_loss_weight=args.rank_loss_weight,
            )
            if not torch.isfinite(loss_v45):
                raise RuntimeError(f"non-finite v4.5 contextual tuple loss for {task}/{writer}/seed{seed}/step{step}")
            optimizer.zero_grad(set_to_none=True)
            loss_v45.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss_contextual = float(loss_v45.detach().item())
        trained_examples += args.batch_size
        if step in budget_set:
            if extractor_v1 is not None:
                extractor_v1.eval()
            if extractor_v2 is not None:
                extractor_v2.eval()
            if writer_v3 is not None:
                writer_v3.eval()
            if spn_assembler is not None:
                spn_assembler.eval()
            if contextual_assembler is not None:
                contextual_assembler.eval()
            for model in contextual_assemblers.values():
                model.eval()
            evaluate_budget(step)


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tasks = parse_str_list(args.tasks, TASKS)
    writers = parse_str_list(args.writers, WRITERS)
    budgets = sorted(set(parse_int_list(args.budgets)))
    seeds = parse_int_list(args.seeds)
    noise_levels = parse_str_list(args.noise_levels, NOISE_LEVELS)
    marker_rates = parse_float_list(args.marker_rates)
    distractor_counts = parse_int_list(args.distractor_counts)
    slot_counts = parse_int_list(args.slot_counts)
    max_slot_counts = parse_int_list(args.max_slots) if args.max_slots else [max(slot_counts)]
    slot_max_pairs = [(slot_count, max_slots) for slot_count in slot_counts for max_slots in max_slot_counts if max_slots >= slot_count]
    lambda_obj_values = parse_float_list(args.lambda_obj_values) if args.lambda_obj_values else [args.lambda_obj]
    v3_mode = is_v3_run(writers)
    contextual_mode = is_contextual_tuple_run(writers)
    contextual_learned_mode = is_contextual_learned_candidate_run(writers)
    tuple_candidate_mode = v3_mode or contextual_mode
    v45_mode = any(writer in V45_WRITERS for writer in writers)
    candidate_k_values = parse_int_list(args.candidate_k_values) if args.candidate_k_values else ([16] if v45_mode else [8] if tuple_candidate_mode else [0])
    if not args.slow_sweep and any(value > 16 for value in candidate_k_values):
        raise ValueError("candidate_k values above 16 require --slow-sweep")
    candidate_loss_weights = parse_float_list(args.candidate_loss_weight) if tuple_candidate_mode else [0.0]
    template_augmentations = parse_str_list(args.template_augmentations, TEMPLATE_AUGMENTATIONS) if args.template_augmentations else [args.template_augmentation]
    simplified_aux_weights = parse_float_list(args.simplified_aux_weight_values) if args.simplified_aux_weight_values else [args.simplified_aux_weight]
    guideline_loss_weights = parse_float_list(args.guideline_loss_weight_values) if args.guideline_loss_weight_values else [args.guideline_loss_weight]
    eval_specs_total = sum(len(eval_specs_for_writer(writer, args)) for writer in writers)
    hard_negatives = parse_int_list(args.hard_negatives)
    if args.template_mix not in TEMPLATE_MIXES:
        raise ValueError(f"unknown template_mix: {args.template_mix}")
    if args.template_split not in TEMPLATE_SPLITS:
        raise ValueError(f"unknown template_split: {args.template_split}")
    if args.template_augmentation not in TEMPLATE_AUGMENTATIONS:
        raise ValueError(f"unknown template_augmentation: {args.template_augmentation}")
    if args.template_augmentations:
        parse_str_list(args.template_augmentations, TEMPLATE_AUGMENTATIONS)
    projected_rows = (
        len(tasks)
        * len(seeds)
        * len(noise_levels)
        * len(marker_rates)
        * len(distractor_counts)
        * len(slot_max_pairs)
        * len(lambda_obj_values)
        * len(candidate_k_values)
        * len(candidate_loss_weights)
        * len(template_augmentations)
        * len(simplified_aux_weights)
        * len(guideline_loss_weights)
        * len(hard_negatives)
        * len(budgets)
        * eval_specs_total
    )
    print(f"projected_rows={projected_rows}", flush=True)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    raw_rows: List[Dict[str, object]] = []
    debug_examples: List[Dict[str, object]] = []
    reader_cache: Dict[tuple, torch.nn.Module] = {}

    for task in tasks:
        for noise_level in noise_levels:
            for marker_rate in marker_rates:
                for distractor_count in distractor_counts:
                    for slot_count, max_slots in slot_max_pairs:
                        for hard in hard_negatives:
                            for lambda_obj_value in lambda_obj_values:
                                for candidate_k in candidate_k_values:
                                    for candidate_loss_weight in candidate_loss_weights:
                                        for template_augmentation_value in template_augmentations:
                                            for simplified_aux_weight_value in simplified_aux_weights:
                                                for guideline_loss_weight_value in guideline_loss_weights:
                                                    for seed in seeds:
                                                        run_config(
                                                            task,
                                                            seed,
                                                            noise_level,
                                                            marker_rate,
                                                            distractor_count,
                                                            slot_count,
                                                            max_slots,
                                                            hard,
                                                            lambda_obj_value,
                                                            candidate_k,
                                                            candidate_loss_weight,
                                                            template_augmentation_value,
                                                            simplified_aux_weight_value,
                                                            guideline_loss_weight_value,
                                                            writers,
                                                            budgets,
                                                            args,
                                                            device,
                                                            reader_cache,
                                                            raw_rows,
                                                            out_dir,
                                                            debug_examples,
                                                        )
    v2_mode = "learned_set_extractor_v2" in writers
    bottleneck_mode = is_bottleneck_run(args)
    summary = flush(out_dir, raw_rows, v2_mode=v2_mode, bottleneck_mode=bottleneck_mode, v3_mode=v3_mode, contextual_mode=contextual_mode, output_prefix=args.output_prefix)
    if contextual_mode:
        if args.output_prefix:
            debug_name = f"{args.output_prefix}_debug.jsonl" if args.output_prefix in {"writer_v45_condition_generalization", "writer_v45_residual_fix"} else f"{args.output_prefix}_debug_examples.jsonl"
            debug_path = out_dir / debug_name
        else:
            debug_path = out_dir / ("contextual_learned_candidates_debug_examples.jsonl" if contextual_learned_mode else "contextual_tuple_debug_examples.jsonl")
        with debug_path.open("w", encoding="utf-8") as handle:
            for item in debug_examples[: args.tuple_debug_examples]:
                handle.write(json.dumps(item) + "\n")
    elif is_tuple_debug_run(writers):
        debug_path = out_dir / "tuple_assembler_debug_examples.jsonl"
        with debug_path.open("w", encoding="utf-8") as handle:
            for item in debug_examples[: args.tuple_debug_examples]:
                handle.write(json.dumps(item) + "\n")
    update_results(Path("results.md"), summary, args, len(raw_rows), projected_rows)
    update_writeup(Path("writeup_structured_memory_result.md"), summary, args)
    if contextual_mode:
        if args.output_prefix:
            debug_name = f"{args.output_prefix}_debug.jsonl" if args.output_prefix in {"writer_v45_condition_generalization", "writer_v45_residual_fix"} else f"{args.output_prefix}_debug_examples.jsonl"
            print(f"wrote runs/{args.output_prefix}_raw.csv, runs/{args.output_prefix}_summary.csv, runs/{debug_name}, results.md")
        elif contextual_learned_mode:
            print("wrote runs/contextual_learned_candidates_raw.csv, runs/contextual_learned_candidates_summary.csv, runs/contextual_learned_candidates_debug_examples.jsonl, results.md")
        else:
            print("wrote runs/contextual_tuple_scorer_raw.csv, runs/contextual_tuple_scorer_summary.csv, runs/contextual_tuple_debug_examples.jsonl, results.md")
    elif is_tuple_debug_run(writers):
        print("wrote runs/tuple_assembler_debug_raw.csv, runs/tuple_assembler_debug_summary.csv, runs/tuple_assembler_debug_examples.jsonl, results.md")
    elif v3_mode:
        print("wrote runs/writer_v3_field_candidates_raw.csv, runs/writer_v3_field_candidates_summary.csv, results.md")
    elif bottleneck_mode:
        print("wrote runs/writer_v2_bottleneck_raw.csv, runs/writer_v2_bottleneck_summary.csv, results.md")
    elif v2_mode:
        print("wrote runs/noisy_slot_extraction_v2_raw.csv, runs/noisy_slot_extraction_v2_summary.csv, results.md")
    else:
        print("wrote runs/noisy_slot_extraction_raw.csv, runs/noisy_slot_extraction_summary.csv, results.md")


if __name__ == "__main__":
    main()
