from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpm_lite.data import FactRecallConfig, FactRecallDataset, VOCAB_SIZE
from hpm_lite.metrics import answer_cross_entropy, answer_span_exact_accuracy, count_parameters
from hpm_lite.model import AnswerTransformerConfig, AnswerTransformerModel
from hpm_lite.noisy_extraction import ContextualTupleEdgeScorer, writer_quality_metrics
from hpm_lite.structured_readout import (
    LearnedConditionReader,
    LearnedSetReader,
    symbolic_condition_binding_metrics,
    symbolic_set_metrics,
)
from hpm_lite.train import TinyAdamW
from hpm_lite.utils import ensure_dir, resolve_device, set_seed


TASKS = {"noisy_conditional", "noisy_coexisting"}
MODELS = {
    "transformer_baseline",
    "integrated_memory_v1",
    "integrated_memory_v1_coexisting_v2",
    "integrated_memory_v1_final",
    "integrated_memory_v1_2_condition_v3_k16",
    "integrated_memory_v1_2_condition_v3_cond24",
    "integrated_memory_v1_2_condition_v3_cond32",
    "integrated_memory_v1_3_condition_adaptive",
    "integrated_memory_v1_final_oracle_slots",
    "integrated_memory_v1_final_oracle_writer_learned_reader",
    "integrated_memory_v1_final_learned_writer_symbolic_reader",
    "integrated_memory_v1_final_oracle_condition_candidates",
    "integrated_memory_v1_final_oracle_key_candidates",
    "integrated_memory_v1_final_oracle_value_candidates",
    "integrated_memory_v1_oracle_slots",
    "integrated_memory_v1_oracle_writer_learned_reader",
    "integrated_memory_v1_learned_writer_symbolic_reader",
}
TEMPLATE_SPLITS = {"heldout", "random"}


RAW_COLUMNS = [
    "task",
    "model",
    "writer_mode",
    "seed",
    "budget_steps",
    "template_split",
    "noise_level",
    "marker_rate",
    "distractor_count",
    "slot_count",
    "max_slots",
    "candidate_k",
    "exact_accuracy",
    "answer_ce",
    "writer_exact",
    "reader_exact",
    "condition_exact",
    "set_exact",
    "set_precision",
    "set_recall",
    "set_f1",
    "slot_f1",
    "key_recall",
    "value_recall",
    "missing_key_rate",
    "missing_condition_rate",
    "missing_value_rate",
    "all_slots_exact",
    "missed_positive_rate",
    "extra_false_positive_rate",
    "oracle_slot_reader_exact",
    "learned_slot_reader_exact",
    "symbolic_reader_exact",
    "condition_recall",
    "condition_candidate_recall",
    "condition_candidate_precision",
    "condition_pool_size",
    "exact_slot_available_rate",
    "all_field_candidate_recall",
    "tuple_scoring_error_rate",
    "reader_symbolic_gap",
    "escalation_rate",
    "false_escalation_rate",
    "missed_escalation_rate",
    "fraction_failures_avoided",
    "train_time",
    "examples_per_sec",
    "inference_examples_per_sec",
    "parameter_count",
    "writer_parameter_count",
    "reader_parameter_count",
    "gpu_memory_mb",
]

SUMMARY_COLUMNS = [
    "task",
    "model",
    "writer_mode",
    "budget_steps",
    "template_split",
    "noise_level",
    "marker_rate",
    "distractor_count",
    "slot_count",
    "max_slots",
    "candidate_k",
    "n",
    "exact_accuracy_mean",
    "exact_accuracy_std",
    "answer_ce_mean",
    "answer_ce_std",
    "writer_exact_mean",
    "writer_exact_std",
    "reader_exact_mean",
    "reader_exact_std",
    "condition_exact_mean",
    "condition_exact_std",
    "set_exact_mean",
    "set_exact_std",
    "set_precision_mean",
    "set_precision_std",
    "set_recall_mean",
    "set_recall_std",
    "set_f1_mean",
    "set_f1_std",
    "slot_f1_mean",
    "slot_f1_std",
    "key_recall_mean",
    "key_recall_std",
    "value_recall_mean",
    "value_recall_std",
    "missing_key_rate_mean",
    "missing_key_rate_std",
    "missing_condition_rate_mean",
    "missing_condition_rate_std",
    "missing_value_rate_mean",
    "missing_value_rate_std",
    "all_slots_exact_mean",
    "all_slots_exact_std",
    "missed_positive_rate_mean",
    "missed_positive_rate_std",
    "extra_false_positive_rate_mean",
    "extra_false_positive_rate_std",
    "oracle_slot_reader_exact_mean",
    "oracle_slot_reader_exact_std",
    "learned_slot_reader_exact_mean",
    "learned_slot_reader_exact_std",
    "symbolic_reader_exact_mean",
    "symbolic_reader_exact_std",
    "condition_recall_mean",
    "condition_recall_std",
    "condition_candidate_recall_mean",
    "condition_candidate_recall_std",
    "condition_candidate_precision_mean",
    "condition_candidate_precision_std",
    "condition_pool_size_mean",
    "condition_pool_size_std",
    "exact_slot_available_rate_mean",
    "exact_slot_available_rate_std",
    "all_field_candidate_recall_mean",
    "all_field_candidate_recall_std",
    "tuple_scoring_error_rate_mean",
    "tuple_scoring_error_rate_std",
    "reader_symbolic_gap_mean",
    "reader_symbolic_gap_std",
    "escalation_rate_mean",
    "escalation_rate_std",
    "false_escalation_rate_mean",
    "false_escalation_rate_std",
    "missed_escalation_rate_mean",
    "missed_escalation_rate_std",
    "fraction_failures_avoided_mean",
    "fraction_failures_avoided_std",
    "train_time_mean",
    "train_time_std",
    "examples_per_sec_mean",
    "examples_per_sec_std",
    "inference_examples_per_sec_mean",
    "inference_examples_per_sec_std",
    "parameter_count_mean",
    "parameter_count_std",
    "writer_parameter_count_mean",
    "writer_parameter_count_std",
    "reader_parameter_count_mean",
    "reader_parameter_count_std",
    "gpu_memory_mb_mean",
    "gpu_memory_mb_std",
    "memory_model_gain_over_transformer",
    "slowdown_vs_transformer",
]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str, valid: set[str] | None = None) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if valid is not None:
        unknown = [item for item in items if item not in valid]
        if unknown:
            raise ValueError(f"unknown values {unknown}; expected one of {sorted(valid)}")
    return items


def task_has_condition(task: str) -> bool:
    return task == "noisy_conditional"


def make_dataset(
    task: str,
    args: argparse.Namespace,
    seed: int,
    noise_level: str,
    marker_rate: float,
    distractor_count: int,
    slot_count: int,
    split_phase: str,
    template_split: str,
    template_augmentation: str = "none",
) -> FactRecallDataset:
    template_mix = args.template_mix
    if template_split == "heldout":
        template_mix = "simple" if split_phase == "train" else "paraphrase"
    return FactRecallDataset(
        FactRecallConfig(
            seq_len=args.seq_len,
            window=args.window,
            task=task,
            seed=seed,
            num_facts=slot_count,
            oracle_memory=True,
            noise_level=noise_level,
            marker_rate=marker_rate,
            distractor_count=distractor_count,
            template_mix=template_mix,
            template_augmentation=template_augmentation if split_phase == "train" else "none",
        )
    )


def make_reader(task: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    kwargs = {
        "reader_dim": args.d_model,
        "hidden": args.reader_hidden,
        "layers": 2,
        "dropout": 0.0,
        "train_embeddings": True,
    }
    if task_has_condition(task):
        return LearnedConditionReader(**kwargs).to(device)
    return LearnedSetReader(**kwargs).to(device)


def train_reader(
    task: str,
    args: argparse.Namespace,
    dataset: FactRecallDataset,
    device: torch.device,
) -> torch.nn.Module:
    reader = make_reader(task, args, device)
    optimizer = TinyAdamW(reader.parameters(), lr=args.lr, weight_decay=0.0)
    reader.train()
    for _ in range(args.reader_pretrain_steps):
        batch = dataset.sample_batch(args.batch_size, device=device)
        loss = reader.loss(batch, "normal")
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite reader pretrain loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        optimizer.step()
    reader.eval()
    if args.writer_mode == "frozen":
        for parameter in reader.parameters():
            parameter.requires_grad_(False)
    return reader


def make_writer(
    task: str,
    args: argparse.Namespace,
    device: torch.device,
    coexisting_v2: bool = False,
    condition_variant: str = "full",
) -> ContextualTupleEdgeScorer:
    proposer_variant = "coexisting_full" if coexisting_v2 else condition_variant
    return ContextualTupleEdgeScorer(
        max_slots=args.max_slots,
        seq_len=args.seq_len,
        has_condition=task_has_condition(task),
        extractor_dim=args.d_model,
        hidden=args.extractor_hidden,
        layers=args.layers,
        dropout=0.0,
        condition_proposer_variant=proposer_variant,
        simplified_aux_weight=args.simplified_aux_weight,
        guideline_loss_weight=args.guideline_loss_weight,
    ).to(device)


def candidate_k_for_model(args: argparse.Namespace, task: str, model_name: str) -> int | List[int]:
    if not task_has_condition(task):
        return [args.key_candidate_k, args.value_candidate_k]
    if model_name == "integrated_memory_v1_3_condition_adaptive":
        return [args.key_candidate_k, args.value_candidate_k, args.condition_candidate_k]
    if model_name.endswith("_k16"):
        condition_k = 16
    elif model_name.endswith("_cond24"):
        condition_k = 24
    elif model_name.endswith("_cond32"):
        condition_k = 32
    elif model_name.startswith("integrated_memory_v1_2_condition_v3"):
        condition_k = args.condition_candidate_k
    else:
        return args.candidate_k
    return [args.key_candidate_k, args.value_candidate_k, condition_k]


def candidate_k_for_training(args: argparse.Namespace, task: str) -> int | List[int]:
    if not task_has_condition(task):
        return [args.key_candidate_k, args.value_candidate_k]
    return [args.key_candidate_k, args.value_candidate_k, args.condition_candidate_k]


def candidate_k_label(candidate_k: int | List[int]) -> str | int:
    if isinstance(candidate_k, list):
        return "/".join(str(value) for value in candidate_k)
    return candidate_k


def memory_reader_exact(task: str, reader: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> float:
    metrics = reader.metrics(batch, "normal")
    if task_has_condition(task):
        return float(metrics.get("learned_condition_exact", 0.0))
    return float(metrics.get("learned_set_exact", 0.0))


def memory_reader_ce(task: str, reader: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not task_has_condition(task):
        return reader.loss(batch, "normal")
    scores = reader.slot_scores(batch, "normal")
    values_pos = batch["memory_token_positions"][:, :, 1].clamp_min(0)
    row = torch.arange(batch["input_ids"].size(0), device=batch["input_ids"].device)[:, None]
    values = batch["input_ids"][row, values_pos]
    vocab_logits = scores.new_full((scores.size(0), VOCAB_SIZE), -20.0)
    for b in range(scores.size(0)):
        for slot in range(scores.size(1)):
            if not bool(batch["memory_mask"][b, slot].item()):
                continue
            token = int(values[b, slot].item())
            vocab_logits[b, token] = torch.maximum(vocab_logits[b, token], scores[b, slot])
    return F.cross_entropy(vocab_logits, batch["answer_tokens"])


def evaluate_transformer(
    model: AnswerTransformerModel,
    dataset: FactRecallDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ce_sum = 0.0
    exact_sum = 0.0
    total = 0
    eval_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            batch = dataset.sample_batch(args.batch_size, device=device)
            logits = model(batch["input_ids"])
            ce = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
            exact = answer_span_exact_accuracy(logits, batch["target_ids"], batch["loss_mask"])
            ce_sum += float(ce.item()) * args.batch_size
            exact_sum += float(exact.item()) * args.batch_size
            total += args.batch_size
    eval_time = time.perf_counter() - eval_start
    return {
        "exact_accuracy": exact_sum / max(total, 1),
        "inference_examples_per_sec": total / max(eval_time, 1.0e-9),
        "answer_ce": ce_sum / max(total, 1),
        "writer_exact": "",
        "reader_exact": "",
        "condition_exact": "",
        "set_exact": "",
        "set_precision": "",
        "set_recall": "",
        "set_f1": "",
        "slot_f1": "",
        "key_recall": "",
        "value_recall": "",
        "missing_key_rate": "",
        "missing_condition_rate": "",
        "missing_value_rate": "",
        "all_slots_exact": "",
        "missed_positive_rate": "",
        "extra_false_positive_rate": "",
        "oracle_slot_reader_exact": "",
        "learned_slot_reader_exact": "",
        "symbolic_reader_exact": "",
        "condition_recall": "",
        "condition_candidate_recall": "",
        "condition_candidate_precision": "",
        "condition_pool_size": "",
        "exact_slot_available_rate": "",
        "all_field_candidate_recall": "",
        "tuple_scoring_error_rate": "",
        "reader_symbolic_gap": "",
        "escalation_rate": "",
        "false_escalation_rate": "",
        "missed_escalation_rate": "",
        "fraction_failures_avoided": "",
    }


def candidate_mode_for_model(model_name: str) -> str:
    if model_name.endswith("_oracle_condition_candidates"):
        return "oracle_cond_candidates"
    if model_name.endswith("_oracle_key_candidates"):
        return "oracle_key_candidates"
    if model_name.endswith("_oracle_value_candidates"):
        return "oracle_value_candidates"
    return "learned_candidates"


def _merge_written_batches(
    base: Dict[str, torch.Tensor],
    replacement: Dict[str, torch.Tensor],
    replace_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for key, value in base.items():
        if key in {"candidate_positions", "candidate_mask"}:
            continue
        if isinstance(value, torch.Tensor) and key in replacement and replacement[key].shape[:1] == value.shape[:1]:
            merged[key] = value.clone()
            merged[key][replace_mask] = replacement[key][replace_mask]
        else:
            merged[key] = value
    for key in ("candidate_positions", "candidate_mask"):
        if key in replacement and key in base:
            base_value = base[key]
            repl_value = replacement[key]
            if isinstance(base_value, torch.Tensor) and isinstance(repl_value, torch.Tensor) and repl_value.shape[2] >= base_value.shape[2]:
                padded = repl_value.clone()
                padded[~replace_mask, :, : base_value.shape[2]] = base_value[~replace_mask]
                merged[key] = padded
    return merged


def _condition_confidence_signals(
    writer: ContextualTupleEdgeScorer,
    reader: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    written: Dict[str, torch.Tensor],
    candidate_k: int | List[int],
) -> Dict[str, torch.Tensor]:
    scores, _, tuple_mask, candidate_positions, candidate_mask = writer.score_tuples(
        batch,
        candidate_k=candidate_k,
        candidate_mode="learned_candidates",
    )
    logits = writer._last_candidate_logits
    bsz = batch["input_ids"].size(0)
    device = batch["input_ids"].device
    condition_top = torch.full((bsz,), -1.0e9, device=device)
    condition_margin = torch.zeros((bsz,), device=device)
    condition_entropy = torch.zeros((bsz,), device=device)
    if logits is not None:
        for b in range(bsz):
            valid = candidate_mask[b, 2]
            if not bool(valid.any().item()):
                continue
            positions = candidate_positions[b, 2, valid]
            values = logits[b, 2, positions]
            sorted_values = torch.sort(values, descending=True).values
            condition_top[b] = sorted_values[0]
            condition_margin[b] = sorted_values[0] - (sorted_values[1] if sorted_values.numel() > 1 else sorted_values[0])
            probs = torch.softmax(values, dim=0)
            condition_entropy[b] = -(probs * torch.log(probs.clamp_min(1.0e-9))).sum()
    tuple_margin = torch.zeros((bsz,), device=device)
    for b in range(bsz):
        valid_scores = scores[b, tuple_mask[b]]
        if valid_scores.numel() == 0:
            continue
        sorted_scores = torch.sort(valid_scores, descending=True).values
        tuple_margin[b] = sorted_scores[0] - (sorted_scores[1] if sorted_scores.numel() > 1 else sorted_scores[0])
    reader_margin = torch.zeros((bsz,), device=device)
    if isinstance(reader, LearnedConditionReader):
        reader_scores = reader.slot_scores(written, "normal")
        for b in range(bsz):
            valid = written["memory_mask"][b]
            valid_scores = reader_scores[b, valid]
            if valid_scores.numel() == 0:
                continue
            sorted_scores = torch.sort(valid_scores, descending=True).values
            reader_margin[b] = sorted_scores[0] - (sorted_scores[1] if sorted_scores.numel() > 1 else sorted_scores[0])
    return {
        "condition_top": condition_top,
        "condition_margin": condition_margin,
        "condition_entropy": condition_entropy,
        "tuple_margin": tuple_margin,
        "reader_margin": reader_margin,
    }


def adaptive_condition_predict(
    writer: ContextualTupleEdgeScorer,
    reader: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    k16 = [args.key_candidate_k, args.value_candidate_k, 16]
    k32 = [args.key_candidate_k, args.value_candidate_k, 32]
    written16 = writer.predict_batch(
        batch,
        candidate_k=k16,
        candidate_mode="learned_candidates",
        decode_mode="top_true_count",
        tuple_pruning="none",
    )
    signals = _condition_confidence_signals(writer, reader, batch, written16, k16)
    risk = torch.zeros(batch["input_ids"].size(0), dtype=torch.bool, device=batch["input_ids"].device)
    if args.adaptive_use_condition_margin:
        risk |= signals["condition_margin"] < args.adaptive_condition_margin_threshold
    if args.adaptive_use_condition_top:
        risk |= signals["condition_top"] < args.adaptive_condition_top_threshold
    if args.adaptive_use_condition_entropy:
        risk |= signals["condition_entropy"] > args.adaptive_condition_entropy_threshold
    if args.adaptive_use_tuple_margin:
        risk |= signals["tuple_margin"] < args.adaptive_tuple_margin_threshold
    if args.adaptive_use_reader_confidence:
        risk |= signals["reader_margin"] < args.adaptive_reader_margin_threshold

    if bool(risk.any().item()):
        written32 = writer.predict_batch(
            batch,
            candidate_k=k32,
            candidate_mode="learned_candidates",
            decode_mode="top_true_count",
            tuple_pruning="none",
        )
        written = _merge_written_batches(written16, written32, risk)
    else:
        written32 = None
        written = written16

    with torch.no_grad():
        metrics16 = symbolic_condition_binding_metrics(written16, "normal")
        final_metrics = symbolic_condition_binding_metrics(written, "normal")
        exact16 = torch.zeros_like(risk, dtype=torch.bool)
        exact_final = torch.zeros_like(risk, dtype=torch.bool)
        if isinstance(reader, LearnedConditionReader):
            scores16 = reader.slot_scores(written16, "normal")
            scores_final = reader.slot_scores(written, "normal")
            target = batch["answer_tokens"]
            for b in range(batch["input_ids"].size(0)):
                for scores_tensor, written_batch, out in ((scores16, written16, exact16), (scores_final, written, exact_final)):
                    if not bool(written_batch["memory_mask"][b].any().item()):
                        continue
                    slot = int(scores_tensor[b].argmax().item())
                    if slot >= written_batch["memory_token_positions"].size(1) or not bool(written_batch["memory_mask"][b, slot].item()):
                        continue
                    value_pos = int(written_batch["memory_token_positions"][b, slot, 1].item())
                    if value_pos >= 0 and int(written_batch["input_ids"][b, value_pos].item()) == int(target[b].item()):
                        out[b] = True
        failures16 = ~exact16
        failures_final = ~exact_final
        avoided = failures16 & exact_final
        false_escalation = risk & exact16
        missed_escalation = (~risk) & failures16
    diag = {
        "escalation_rate": float(risk.float().mean().item()),
        "false_escalation_rate": float(false_escalation.float().mean().item()),
        "missed_escalation_rate": float(missed_escalation.float().mean().item()),
        "fraction_failures_avoided": float(avoided.sum().item() / max(int(failures16.sum().item()), 1)),
        "symbolic_k16_exact": float(metrics16.get("condition_symbolic_exact", 0.0)),
        "symbolic_adaptive_exact": float(final_metrics.get("condition_symbolic_exact", 0.0)),
    }
    return written, diag


def evaluate_integrated(
    task: str,
    model_name: str,
    writer: ContextualTupleEdgeScorer,
    reader: torch.nn.Module,
    dataset: FactRecallDataset,
    args: argparse.Namespace,
    device: torch.device,
    candidate_k: int | List[int] | None = None,
) -> Dict[str, float]:
    writer.eval()
    reader.eval()
    sums: Dict[str, float] = {}
    total = 0
    eval_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            batch = dataset.sample_batch(args.batch_size, device=device)
            use_oracle_writer = model_name in {
                "integrated_memory_v1_oracle_slots",
                "integrated_memory_v1_oracle_writer_learned_reader",
                "integrated_memory_v1_final_oracle_slots",
                "integrated_memory_v1_final_oracle_writer_learned_reader",
            }
            use_symbolic_reader = model_name in {
                "integrated_memory_v1_oracle_slots",
                "integrated_memory_v1_learned_writer_symbolic_reader",
                "integrated_memory_v1_final_learned_writer_symbolic_reader",
            }
            candidate_mode = candidate_mode_for_model(model_name)
            eval_candidate_k = candidate_k if candidate_k is not None else args.candidate_k
            adaptive_diag: Dict[str, float] = {}
            if use_oracle_writer:
                written = {key: value.clone() for key, value in batch.items()}
            elif model_name == "integrated_memory_v1_3_condition_adaptive" and task_has_condition(task):
                written, adaptive_diag = adaptive_condition_predict(writer, reader, batch, args)
            else:
                written = writer.predict_batch(
                    batch,
                    candidate_k=eval_candidate_k,
                    candidate_mode=candidate_mode,
                    decode_mode="top_true_count",
                    tuple_pruning="none",
                )
            writer_metrics = (
                {
                    "all_slots_exact": 1.0,
                    "slot_f1": 1.0,
                    "learned_candidate_key_recall": 1.0,
                    "learned_candidate_value_recall": 1.0,
                    "learned_candidate_cond_recall": 1.0,
                    "learned_candidate_all_field_recall": 1.0,
                    "candidate_miss_rate_key": 0.0,
                    "candidate_miss_rate_cond": 0.0,
                    "candidate_miss_rate_value": 0.0,
                    "learned_candidate_cond_precision": 1.0,
                    "candidate_pool_size_condition": 0.0,
                    "exact_slot_available_rate": 1.0,
                    "tuple_scoring_error_rate": 0.0,
                    "missed_positive_rate": 0.0,
                    "extra_false_positive_rate": 0.0,
                }
                if use_oracle_writer
                else writer_quality_metrics(batch, written, task_has_condition(task))
            )
            learned_metrics = reader.metrics(written, "normal")
            oracle_learned_metrics = reader.metrics(batch, "normal")
            if task_has_condition(task):
                symbolic_metrics = symbolic_condition_binding_metrics(written, "normal")
                learned_exact = float(learned_metrics.get("learned_condition_exact", 0.0))
                symbolic_exact = float(symbolic_metrics.get("condition_symbolic_exact", 0.0))
                set_precision = set_recall = set_f1 = ""
            else:
                symbolic_metrics = symbolic_set_metrics(written, "normal")
                learned_exact = float(learned_metrics.get("learned_set_exact", 0.0))
                symbolic_exact = float(symbolic_metrics.get("symbolic_set_exact", 0.0))
                set_precision = float(
                    (symbolic_metrics if use_symbolic_reader else learned_metrics).get(
                        "symbolic_set_precision" if use_symbolic_reader else "learned_set_precision",
                        0.0,
                    )
                )
                set_recall = float(
                    (symbolic_metrics if use_symbolic_reader else learned_metrics).get(
                        "symbolic_set_recall" if use_symbolic_reader else "learned_set_recall",
                        0.0,
                    )
                )
                set_f1 = float(
                    (symbolic_metrics if use_symbolic_reader else learned_metrics).get(
                        "symbolic_set_f1" if use_symbolic_reader else "learned_set_f1",
                        0.0,
                    )
                )
            exact = symbolic_exact if use_symbolic_reader else learned_exact
            ce_value: float | str
            if use_symbolic_reader:
                ce_value = ""
            else:
                ce = memory_reader_ce(task, reader, written)
                ce_value = float(ce.item())
            oracle_slot_reader_exact = float(
                oracle_learned_metrics.get("learned_condition_exact", oracle_learned_metrics.get("learned_set_exact", 0.0))
            )
            metrics = {
                "exact_accuracy": exact,
                "answer_ce": ce_value,
                "writer_exact": writer_metrics.get("all_slots_exact", 0.0),
                "reader_exact": exact,
                "condition_exact": exact if task_has_condition(task) else "",
                "set_exact": exact if not task_has_condition(task) else "",
                "set_precision": set_precision,
                "set_recall": set_recall,
                "set_f1": set_f1,
                "slot_f1": writer_metrics.get("slot_f1", 0.0),
                "key_recall": writer_metrics.get("learned_candidate_key_recall", 1.0),
                "value_recall": writer_metrics.get("learned_candidate_value_recall", 1.0),
                "missing_key_rate": writer_metrics.get("candidate_miss_rate_key", 0.0),
                "missing_condition_rate": writer_metrics.get("candidate_miss_rate_cond", 0.0),
                "missing_value_rate": writer_metrics.get("candidate_miss_rate_value", 0.0),
                "all_slots_exact": writer_metrics.get("all_slots_exact", 0.0),
                "missed_positive_rate": 1.0 - float(set_recall) if set_recall != "" else "",
                "extra_false_positive_rate": 1.0 - float(set_precision) if set_precision != "" else "",
                "oracle_slot_reader_exact": oracle_slot_reader_exact,
                "learned_slot_reader_exact": learned_exact,
                "symbolic_reader_exact": symbolic_exact,
                "condition_recall": writer_metrics.get("learned_candidate_cond_recall", 1.0),
                "condition_candidate_recall": writer_metrics.get("learned_candidate_cond_recall", 1.0),
                "condition_candidate_precision": writer_metrics.get("learned_candidate_cond_precision", 1.0),
                "condition_pool_size": writer_metrics.get("candidate_pool_size_condition", 0.0),
                "exact_slot_available_rate": writer_metrics.get("exact_slot_available_rate", 1.0),
                "all_field_candidate_recall": writer_metrics.get("learned_candidate_all_field_recall", 1.0),
                "tuple_scoring_error_rate": writer_metrics.get("tuple_scoring_error_rate", 0.0),
                "reader_symbolic_gap": symbolic_exact - learned_exact,
                "escalation_rate": adaptive_diag.get("escalation_rate", ""),
                "false_escalation_rate": adaptive_diag.get("false_escalation_rate", ""),
                "missed_escalation_rate": adaptive_diag.get("missed_escalation_rate", ""),
                "fraction_failures_avoided": adaptive_diag.get("fraction_failures_avoided", ""),
            }
            for key, value in metrics.items():
                if value == "":
                    continue
                sums[key] = sums.get(key, 0.0) + float(value) * args.batch_size
            total += args.batch_size
    out = {key: value / max(total, 1) for key, value in sums.items()}
    out["inference_examples_per_sec"] = total / max(time.perf_counter() - eval_start, 1.0e-9)
    return out


def _slot_records(batch: Dict[str, torch.Tensor], b: int, predicted: bool = False) -> List[Dict[str, int]]:
    records: List[Dict[str, int]] = []
    mask = batch["memory_mask"][b]
    positions = batch["memory_token_positions"][b]
    input_ids = batch["input_ids"][b]
    for slot_tensor in torch.nonzero(mask, as_tuple=False).flatten():
        slot = int(slot_tensor.item())
        key_pos = int(positions[slot, 0].item())
        value_pos = int(positions[slot, 1].item())
        records.append(
            {
                "slot": slot,
                "key_pos": key_pos,
                "key_token": int(input_ids[key_pos].item()) if key_pos >= 0 else -1,
                "value_pos": value_pos,
                "value_token": int(input_ids[value_pos].item()) if value_pos >= 0 else -1,
                "predicted": int(predicted),
            }
        )
    return records


def coexisting_failure_examples(
    writer: ContextualTupleEdgeScorer,
    reader: torch.nn.Module,
    dataset: FactRecallDataset,
    args: argparse.Namespace,
    device: torch.device,
    limit: int,
) -> List[Dict[str, object]]:
    if limit <= 0:
        return []
    writer.eval()
    reader.eval()
    examples: List[Dict[str, object]] = []
    attempts = 0
    with torch.no_grad():
        while len(examples) < limit and attempts < max(20, limit * 20):
            attempts += 1
            batch = dataset.sample_batch(args.batch_size, device=device)
            written = writer.predict_batch(
                batch,
                candidate_k=args.candidate_k,
                candidate_mode="learned_candidates",
                decode_mode="top_true_count",
                tuple_pruning="none",
            )
            logits = reader.slot_logits(written, "normal")
            selected = (torch.sigmoid(logits) >= reader.threshold) & written["memory_mask"]
            scores, tuple_positions, tuple_mask, candidate_positions, candidate_mask = writer.score_tuples(
                batch,
                candidate_k=args.candidate_k,
                candidate_mode="learned_candidates",
            )
            for b in range(batch["input_ids"].size(0)):
                answer_mask = batch["answer_target_mask"][b]
                true_answer_set = {
                    int(token.item())
                    for token in batch["answer_token_spans"][b, answer_mask]
                }
                pred_values = []
                for slot_tensor in torch.nonzero(selected[b], as_tuple=False).flatten():
                    slot = int(slot_tensor.item())
                    value_pos = int(written["memory_token_positions"][b, slot, 1].item())
                    if value_pos >= 0:
                        pred_values.append(int(written["input_ids"][b, value_pos].item()))
                predicted_answer_set = set(pred_values)
                if predicted_answer_set == true_answer_set:
                    continue
                gold_key_positions = {
                    int(batch["memory_token_positions"][b, slot, 0].item())
                    for slot in torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
                }
                gold_value_positions = {
                    int(batch["memory_token_positions"][b, slot, 1].item())
                    for slot in torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
                }
                key_candidates = [
                    {
                        "index": idx,
                        "pos": int(pos.item()),
                        "token": int(batch["input_ids"][b, pos].item()),
                    }
                    for idx, pos in enumerate(candidate_positions[b, 0])
                    if bool(candidate_mask[b, 0, idx].item())
                ]
                value_candidates = [
                    {
                        "index": idx,
                        "pos": int(pos.item()),
                        "token": int(batch["input_ids"][b, pos].item()),
                    }
                    for idx, pos in enumerate(candidate_positions[b, 1])
                    if bool(candidate_mask[b, 1, idx].item())
                ]
                key_candidate_positions = {item["pos"] for item in key_candidates}
                value_candidate_positions = {item["pos"] for item in value_candidates}
                missing_values = sorted(true_answer_set - predicted_answer_set)
                extra_values = sorted(predicted_answer_set - true_answer_set)
                if not gold_key_positions <= key_candidate_positions:
                    category = "missing_key"
                elif not gold_value_positions <= value_candidate_positions:
                    category = "missing_value"
                elif missing_values and not extra_values:
                    category = "set_reader_miss"
                elif extra_values and not missing_values:
                    category = "extra_false_positive"
                elif missing_values or extra_values:
                    category = "wrong_key_value_binding"
                else:
                    category = "integration_bug"
                valid_indices = torch.nonzero(tuple_mask[b], as_tuple=False).flatten()
                ranked = valid_indices[torch.argsort(scores[b, valid_indices], descending=True)]
                top_tuples = []
                for flat_idx_tensor in ranked[:10]:
                    flat_idx = int(flat_idx_tensor.item())
                    fields = [int(x) for x in tuple_positions[b, flat_idx, :2].tolist()]
                    top_tuples.append(
                        {
                            "flat_index": flat_idx,
                            "positions": fields,
                            "tokens": [int(batch["input_ids"][b, pos].item()) if pos >= 0 else -1 for pos in fields],
                            "score": float(scores[b, flat_idx].item()),
                            "prob": float(torch.sigmoid(scores[b, flat_idx]).item()),
                        }
                    )
                examples.append(
                    {
                        "token_sequence": batch["input_ids"][b].tolist(),
                        "true_slots": _slot_records(batch, b, predicted=False),
                        "predicted_slots": _slot_records(written, b, predicted=True),
                        "true_answer_set": sorted(true_answer_set),
                        "predicted_answer_set": sorted(predicted_answer_set),
                        "missing_values": missing_values,
                        "extra_values": extra_values,
                        "key_candidates": key_candidates,
                        "value_candidates": value_candidates,
                        "top_tuple_scores": top_tuples,
                        "failure_category": category,
                    }
                )
                if len(examples) >= limit:
                    break
    return examples


def conditional_failure_examples(
    writer: ContextualTupleEdgeScorer,
    reader: LearnedConditionReader,
    dataset: FactRecallDataset,
    args: argparse.Namespace,
    device: torch.device,
    limit: int,
    candidate_k: int | List[int] | None = None,
) -> List[Dict[str, object]]:
    if limit <= 0:
        return []
    writer.eval()
    reader.eval()
    examples: List[Dict[str, object]] = []
    attempts = 0
    with torch.no_grad():
        while len(examples) < limit and attempts < max(50, limit * 25):
            attempts += 1
            batch = dataset.sample_batch(args.batch_size, device=device)
            eval_candidate_k = candidate_k if candidate_k is not None else args.candidate_k
            written = writer.predict_batch(
                batch,
                candidate_k=eval_candidate_k,
                candidate_mode="learned_candidates",
                decode_mode="top_true_count",
                tuple_pruning="none",
            )
            reader_scores = reader.slot_scores(written, "normal")
            chosen_slots = reader_scores.argmax(dim=1)
            scores, tuple_positions, tuple_mask, candidate_positions, candidate_mask = writer.score_tuples(
                batch,
                candidate_k=eval_candidate_k,
                candidate_mode="learned_candidates",
            )
            input_ids = batch["input_ids"]
            for b in range(batch["input_ids"].size(0)):
                true_slot = int(batch["positive_memory_indices"][b].item())
                true_key_pos = int(batch["memory_token_positions"][b, true_slot, 0].item())
                true_value_pos = int(batch["memory_token_positions"][b, true_slot, 1].item())
                true_cond_pos = int(batch["memory_condition_positions"][b, true_slot].item())
                true_key = int(input_ids[b, true_key_pos].item())
                true_value = int(input_ids[b, true_value_pos].item())
                true_cond = int(input_ids[b, true_cond_pos].item())

                chosen = int(chosen_slots[b].item())
                if chosen >= 0 and chosen < written["memory_mask"].size(1) and bool(written["memory_mask"][b, chosen].item()):
                    pred_key_pos = int(written["memory_token_positions"][b, chosen, 0].item())
                    pred_value_pos = int(written["memory_token_positions"][b, chosen, 1].item())
                    pred_cond_pos = int(written["memory_condition_positions"][b, chosen].item())
                    pred_key = int(input_ids[b, pred_key_pos].item()) if pred_key_pos >= 0 else -1
                    pred_value = int(input_ids[b, pred_value_pos].item()) if pred_value_pos >= 0 else -1
                    pred_cond = int(input_ids[b, pred_cond_pos].item()) if pred_cond_pos >= 0 else -1
                else:
                    pred_key_pos = pred_value_pos = pred_cond_pos = -1
                    pred_key = pred_value = pred_cond = -1

                exact = pred_key == true_key and pred_cond == true_cond and pred_value == true_value
                if exact:
                    continue

                def _candidates(field: int) -> List[Dict[str, int]]:
                    return [
                        {
                            "index": idx,
                            "pos": int(pos.item()),
                            "token": int(input_ids[b, pos].item()),
                        }
                        for idx, pos in enumerate(candidate_positions[b, field])
                        if bool(candidate_mask[b, field, idx].item())
                    ]

                key_candidates = _candidates(0)
                value_candidates = _candidates(1)
                cond_candidates = _candidates(2)
                key_positions = {item["pos"] for item in key_candidates}
                value_positions = {item["pos"] for item in value_candidates}
                cond_positions = {item["pos"] for item in cond_candidates}
                exact_slot_existed = (
                    true_key_pos in key_positions
                    and true_cond_pos in cond_positions
                    and true_value_pos in value_positions
                )
                pred_slot_tuples = set()
                for slot_tensor in torch.nonzero(written["memory_mask"][b], as_tuple=False).flatten():
                    slot = int(slot_tensor.item())
                    key_pos = int(written["memory_token_positions"][b, slot, 0].item())
                    value_pos = int(written["memory_token_positions"][b, slot, 1].item())
                    cond_pos = int(written["memory_condition_positions"][b, slot].item())
                    pred_slot_tuples.add(
                        (
                            int(input_ids[b, key_pos].item()) if key_pos >= 0 else -1,
                            int(input_ids[b, cond_pos].item()) if cond_pos >= 0 else -1,
                            int(input_ids[b, value_pos].item()) if value_pos >= 0 else -1,
                        )
                    )
                true_tuple = (true_key, true_cond, true_value)
                if true_key_pos not in key_positions:
                    category = "missing_key"
                elif true_cond_pos not in cond_positions:
                    category = "missing_condition"
                elif true_value_pos not in value_positions:
                    category = "missing_value"
                elif true_tuple not in pred_slot_tuples:
                    category = "wrong_tuple_binding"
                elif pred_key != true_key or pred_cond != true_cond or pred_value != true_value:
                    category = "reader_error"
                else:
                    category = "integration_bug"

                valid_indices = torch.nonzero(tuple_mask[b], as_tuple=False).flatten()
                ranked = valid_indices[torch.argsort(scores[b, valid_indices], descending=True)]
                top_tuples = []
                for flat_idx_tensor in ranked[:10]:
                    flat_idx = int(flat_idx_tensor.item())
                    fields = [int(x) for x in tuple_positions[b, flat_idx, :3].tolist()]
                    top_tuples.append(
                        {
                            "flat_index": flat_idx,
                            "positions": {
                                "key": fields[0],
                                "condition": fields[2],
                                "value": fields[1],
                            },
                            "tokens": {
                                "key": int(input_ids[b, fields[0]].item()) if fields[0] >= 0 else -1,
                                "condition": int(input_ids[b, fields[2]].item()) if fields[2] >= 0 else -1,
                                "value": int(input_ids[b, fields[1]].item()) if fields[1] >= 0 else -1,
                            },
                            "score": float(scores[b, flat_idx].item()),
                            "prob": float(torch.sigmoid(scores[b, flat_idx]).item()),
                            "is_gold": fields[0] == true_key_pos and fields[1] == true_value_pos and fields[2] == true_cond_pos,
                        }
                    )

                examples.append(
                    {
                        "token_sequence": input_ids[b].tolist(),
                        "true_key": {"pos": true_key_pos, "token": true_key},
                        "true_condition": {"pos": true_cond_pos, "token": true_cond},
                        "true_value": {"pos": true_value_pos, "token": true_value},
                        "predicted_key": {"pos": pred_key_pos, "token": pred_key},
                        "predicted_condition": {"pos": pred_cond_pos, "token": pred_cond},
                        "predicted_value": {"pos": pred_value_pos, "token": pred_value},
                        "candidate_key_list": key_candidates,
                        "candidate_condition_list": cond_candidates,
                        "candidate_value_list": value_candidates,
                        "top_tuple_scores": top_tuples,
                        "exact_slot_existed_in_candidates": exact_slot_existed,
                        "failure_category": category,
                    }
                )
                if len(examples) >= limit:
                    break
    return examples


def train_transformer_config(
    task: str,
    seed: int,
    template_split: str,
    noise_level: str,
    marker_rate: float,
    distractor_count: int,
    slot_count: int,
    budgets: List[int],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, object]]:
    set_seed(seed)
    train_dataset = make_dataset(
        task,
        args,
        seed + 1_000,
        noise_level,
        marker_rate,
        distractor_count,
        slot_count,
        "train",
        template_split,
        args.template_augmentation,
    )
    model = AnswerTransformerModel(
        AnswerTransformerConfig(
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            window=args.window,
            max_seq_len=args.seq_len,
            dropout=0.0,
        )
    ).to(device)
    optimizer = TinyAdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    budget_set = set(budgets)
    rows: List[Dict[str, object]] = []
    start = time.perf_counter()
    trained_examples = 0
    for step in range(1, max(budgets) + 1):
        model.train()
        batch = train_dataset.sample_batch(args.batch_size, device=device)
        logits = model(batch["input_ids"])
        loss = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite transformer loss for {task}/seed{seed}/step{step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        trained_examples += args.batch_size
        if step in budget_set:
            train_time = time.perf_counter() - start
            eval_dataset = make_dataset(
                task,
                args,
                900_000 + seed + step,
                noise_level,
                marker_rate,
                distractor_count,
                slot_count,
                "eval",
                template_split,
            )
            metrics = evaluate_transformer(model, eval_dataset, args, device)
            rows.append(
                base_row(
                    task,
                    "transformer_baseline",
                    seed,
                    step,
                    template_split,
                    noise_level,
                    marker_rate,
                    distractor_count,
                    slot_count,
                    args,
                    train_time,
                    trained_examples,
                    count_parameters(model),
                    0,
                    0,
                    device,
                    metrics,
                )
            )
    return rows


def train_integrated_config(
    task: str,
    seed: int,
    template_split: str,
    noise_level: str,
    marker_rate: float,
    distractor_count: int,
    slot_count: int,
    budgets: List[int],
    memory_models: List[str],
    args: argparse.Namespace,
    device: torch.device,
    debug_examples: List[Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    set_seed(seed + 50_000)
    train_dataset = make_dataset(
        task,
        args,
        seed + 51_000,
        noise_level,
        marker_rate,
        distractor_count,
        slot_count,
        "train",
        template_split,
        "none",
    )
    train_dataset_augmented = make_dataset(
        task,
        args,
        seed + 52_000,
        noise_level,
        marker_rate,
        distractor_count,
        slot_count,
        "train",
        template_split,
        args.template_augmentation,
    )
    reader = train_reader(task, args, train_dataset, device)
    needs_baseline = any(
        model in {"integrated_memory_v1", "integrated_memory_v1_learned_writer_symbolic_reader"}
        for model in memory_models
    )
    needs_v46 = "integrated_memory_v1_coexisting_v2" in memory_models
    final_mode_names = {model for model in memory_models if model.startswith("integrated_memory_v1_final")}
    needs_final = bool(final_mode_names)
    v12_mode_names = {
        model
        for model in memory_models
        if model.startswith("integrated_memory_v1_2_condition_v3") or model == "integrated_memory_v1_3_condition_adaptive"
    }
    needs_v12 = bool(v12_mode_names)
    symbolic_uses_v46 = needs_v46 and "integrated_memory_v1_learned_writer_symbolic_reader" in memory_models
    baseline_writer = make_writer(task, args, device, coexisting_v2=False) if needs_baseline else None
    v46_writer = make_writer(task, args, device, coexisting_v2=True) if needs_v46 or symbolic_uses_v46 else None
    final_writer = (
        make_writer(task, args, device, coexisting_v2=not task_has_condition(task))
        if needs_final
        else None
    )
    v12_writer = (
        make_writer(task, args, device, coexisting_v2=False, condition_variant="condition_v3")
        if needs_v12
        else None
    )
    optimizer_baseline = TinyAdamW(baseline_writer.parameters(), lr=args.lr, weight_decay=0.0) if baseline_writer is not None else None
    optimizer_v46 = TinyAdamW(v46_writer.parameters(), lr=args.lr, weight_decay=0.0) if v46_writer is not None else None
    optimizer_final = TinyAdamW(final_writer.parameters(), lr=args.lr, weight_decay=0.0) if final_writer is not None else None
    optimizer_v12 = TinyAdamW(v12_writer.parameters(), lr=args.lr, weight_decay=0.0) if v12_writer is not None else None
    budget_set = set(budgets)
    rows: List[Dict[str, object]] = []
    start = time.perf_counter()
    trained_examples = 0
    for step in range(1, max(budgets) + 1):
        if baseline_writer is not None and optimizer_baseline is not None:
            baseline_writer.train()
            batch = train_dataset.sample_batch(args.batch_size, device=device)
            loss = baseline_writer.loss(
                batch,
                candidate_k=args.candidate_k,
                candidate_loss_weight=args.candidate_loss_weight,
                tuple_loss_weight=args.tuple_loss_weight,
                rank_loss_weight=args.rank_loss_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite integrated baseline loss for {task}/seed{seed}/step{step}")
            optimizer_baseline.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_writer.parameters(), 1.0)
            optimizer_baseline.step()
        if v46_writer is not None and optimizer_v46 is not None:
            v46_writer.train()
            batch = train_dataset_augmented.sample_batch(args.batch_size, device=device)
            loss = v46_writer.loss(
                batch,
                candidate_k=args.candidate_k,
                candidate_loss_weight=args.candidate_loss_weight,
                tuple_loss_weight=args.tuple_loss_weight,
                rank_loss_weight=args.rank_loss_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite integrated v4.6 loss for {task}/seed{seed}/step{step}")
            optimizer_v46.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(v46_writer.parameters(), 1.0)
            optimizer_v46.step()
        if final_writer is not None and optimizer_final is not None:
            final_writer.train()
            batch = train_dataset_augmented.sample_batch(args.batch_size, device=device)
            loss = final_writer.loss(
                batch,
                candidate_k=args.candidate_k,
                candidate_loss_weight=args.candidate_loss_weight,
                tuple_loss_weight=args.tuple_loss_weight,
                rank_loss_weight=args.rank_loss_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite integrated final loss for {task}/seed{seed}/step{step}")
            optimizer_final.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_writer.parameters(), 1.0)
            optimizer_final.step()
        if v12_writer is not None and optimizer_v12 is not None:
            v12_writer.train()
            batch = train_dataset_augmented.sample_batch(args.batch_size, device=device)
            loss = v12_writer.loss(
                batch,
                candidate_k=candidate_k_for_training(args, task),
                candidate_loss_weight=args.candidate_loss_weight,
                tuple_loss_weight=args.tuple_loss_weight,
                rank_loss_weight=args.rank_loss_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite integrated v1.2 condition v3 loss for {task}/seed{seed}/step{step}")
            optimizer_v12.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(v12_writer.parameters(), 1.0)
            optimizer_v12.step()
        trained_examples += args.batch_size
        if step in budget_set:
            train_time = time.perf_counter() - start
            eval_dataset = make_dataset(
                task,
                args,
                950_000 + seed + step,
                noise_level,
                marker_rate,
                distractor_count,
                slot_count,
                "eval",
                template_split,
            )
            for model_name in memory_models:
                if model_name.startswith("integrated_memory_v1_2_condition_v3") or model_name == "integrated_memory_v1_3_condition_adaptive":
                    writer_for_eval = v12_writer
                elif model_name.startswith("integrated_memory_v1_final"):
                    writer_for_eval = final_writer
                elif model_name == "integrated_memory_v1_coexisting_v2":
                    writer_for_eval = v46_writer
                elif model_name == "integrated_memory_v1_learned_writer_symbolic_reader" and symbolic_uses_v46:
                    writer_for_eval = v46_writer
                else:
                    writer_for_eval = baseline_writer or v46_writer
                if writer_for_eval is None:
                    continue
                eval_candidate_k = candidate_k_for_model(args, task, model_name)
                metrics = evaluate_integrated(task, model_name, writer_for_eval, reader, eval_dataset, args, device, eval_candidate_k)
                writer_params = count_parameters(writer_for_eval)
                rows.append(
                    base_row(
                        task,
                        model_name,
                        seed,
                        step,
                        template_split,
                        noise_level,
                        marker_rate,
                        distractor_count,
                        slot_count,
                        args,
                        train_time,
                        trained_examples,
                        writer_params + count_parameters(reader),
                        writer_params,
                        count_parameters(reader),
                        device,
                        metrics,
                        candidate_k_label(eval_candidate_k),
                    )
                )
            if (
                debug_examples is not None
                and task == "noisy_coexisting"
                and template_split == "heldout"
                and step == max(budgets)
                and (
                    "integrated_memory_v1_final" in memory_models
                    or "integrated_memory_v1_coexisting_v2" in memory_models
                    or "integrated_memory_v1" in memory_models
                )
                and len(debug_examples) < args.debug_examples
            ):
                debug_writer = final_writer or v46_writer or baseline_writer
                if debug_writer is None:
                    continue
                debug_examples.extend(
                    coexisting_failure_examples(
                        debug_writer,
                        reader,
                        eval_dataset,
                        args,
                        device,
                        args.debug_examples - len(debug_examples),
                    )
                )
            if (
                debug_examples is not None
                and args.output_prefix in {"conditional_stress_diagnostic", "conditional_candidate_v12", "conditional_adaptive_v13"}
                and task == "noisy_conditional"
                and template_split == "random"
                and step == max(budgets)
                and (final_writer is not None or v12_writer is not None)
                and isinstance(reader, LearnedConditionReader)
                and len(debug_examples) < args.debug_examples
            ):
                debug_examples.extend(
                    conditional_failure_examples(
                        v12_writer if args.output_prefix in {"conditional_candidate_v12", "conditional_adaptive_v13"} and v12_writer is not None else final_writer,
                        reader,
                        eval_dataset,
                        args,
                        device,
                        args.debug_examples - len(debug_examples),
                        candidate_k_for_training(args, task) if args.output_prefix in {"conditional_candidate_v12", "conditional_adaptive_v13"} else None,
                    )
                )
    return rows


def base_row(
    task: str,
    model: str,
    seed: int,
    budget: int,
    template_split: str,
    noise_level: str,
    marker_rate: float,
    distractor_count: int,
    slot_count: int,
    args: argparse.Namespace,
    train_time: float,
    trained_examples: int,
    parameter_count: int,
    writer_parameter_count: int,
    reader_parameter_count: int,
    device: torch.device,
    metrics: Dict[str, float | str],
    candidate_k_value: str | int | None = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "task": task,
        "model": model,
        "writer_mode": args.writer_mode if model.startswith("integrated_memory_v1") else "",
        "seed": seed,
        "budget_steps": budget,
        "template_split": template_split,
        "noise_level": noise_level,
        "marker_rate": marker_rate,
        "distractor_count": distractor_count,
        "slot_count": slot_count,
        "max_slots": args.max_slots,
        "candidate_k": (candidate_k_value if candidate_k_value is not None else args.candidate_k) if model.startswith("integrated_memory_v1") else "",
        "train_time": train_time,
        "examples_per_sec": trained_examples / max(train_time, 1.0e-9),
        "inference_examples_per_sec": metrics.get("inference_examples_per_sec", ""),
        "parameter_count": parameter_count,
        "writer_parameter_count": writer_parameter_count,
        "reader_parameter_count": reader_parameter_count,
        "gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0,
    }
    row.update(metrics)
    return row


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            row["task"],
            row["model"],
            row.get("writer_mode", ""),
            int(row["budget_steps"]),
            row["template_split"],
            row["noise_level"],
            float(row["marker_rate"]),
            int(row["distractor_count"]),
            int(row["slot_count"]),
            int(row["max_slots"]),
            str(row["candidate_k"]) if row.get("candidate_k", "") != "" else "",
        )
        groups.setdefault(key, []).append(row)
    metrics = [column[:-5] for column in SUMMARY_COLUMNS if column.endswith("_mean")]
    summary: List[Dict[str, object]] = []
    for key in sorted(groups):
        task, model, writer_mode, budget, template_split, noise_level, marker_rate, distractor_count, slot_count, max_slots, candidate_k = key
        group = groups[key]
        out: Dict[str, object] = {
            "task": task,
            "model": model,
            "writer_mode": writer_mode,
            "budget_steps": budget,
            "template_split": template_split,
            "noise_level": noise_level,
            "marker_rate": marker_rate,
            "distractor_count": distractor_count,
            "slot_count": slot_count,
            "max_slots": max_slots,
            "candidate_k": candidate_k if str(model).startswith("integrated_memory_v1") else "",
            "n": len(group),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group if item.get(metric, "") != ""]
            if values:
                out[f"{metric}_mean"] = statistics.mean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
        summary.append(out)
    baseline_lookup = {
        (
            row["task"],
            row["budget_steps"],
            row["template_split"],
            row["noise_level"],
            row["marker_rate"],
            row["distractor_count"],
            row["slot_count"],
            row["max_slots"],
        ): row
        for row in summary
        if row["model"] == "transformer_baseline"
    }
    for row in summary:
        if row["model"] == "transformer_baseline":
            row["memory_model_gain_over_transformer"] = ""
            row["slowdown_vs_transformer"] = ""
            continue
        key = (
            row["task"],
            row["budget_steps"],
            row["template_split"],
            row["noise_level"],
            row["marker_rate"],
            row["distractor_count"],
            row["slot_count"],
            row["max_slots"],
        )
        baseline = baseline_lookup.get(key)
        if baseline is None or baseline.get("exact_accuracy_mean", "") == "" or row.get("exact_accuracy_mean", "") == "":
            row["memory_model_gain_over_transformer"] = ""
        else:
            row["memory_model_gain_over_transformer"] = float(row["exact_accuracy_mean"]) - float(baseline["exact_accuracy_mean"])
        baseline_speed = baseline.get("inference_examples_per_sec_mean", "") if baseline is not None else ""
        row_speed = row.get("inference_examples_per_sec_mean", "")
        if baseline_speed == "" or row_speed == "":
            baseline_speed = baseline.get("examples_per_sec_mean", "") if baseline is not None else ""
            row_speed = row.get("examples_per_sec_mean", "")
        if baseline is None or baseline_speed == "" or row_speed == "":
            row["slowdown_vs_transformer"] = ""
        else:
            row["slowdown_vs_transformer"] = float(baseline_speed) / max(float(row_speed), 1.0e-9)
    return summary


def fmt(value: object) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.4f}"


def update_results(path: Path, summary: List[Dict[str, object]], args: argparse.Namespace, raw_count: int) -> None:
    if args.output_prefix.endswith("_smoke"):
        return

    final_budget = max(parse_int_list(args.budgets))
    rows = [row for row in summary if int(row["budget_steps"]) == final_budget]
    if args.output_prefix == "conditional_adaptive_v13":
        diagnostic_rows = [row for row in rows if row["task"] == "noisy_conditional" and row["template_split"] == "random"]

        def metric(model: str, name: str) -> float | None:
            for row in diagnostic_rows:
                if row["model"] == model:
                    value = row.get(f"{name}_mean", row.get(name, ""))
                    return None if value == "" else float(value)
            return None

        adaptive_exact = metric("integrated_memory_v1_3_condition_adaptive", "exact_accuracy")
        adaptive_recall = metric("integrated_memory_v1_3_condition_adaptive", "condition_candidate_recall")
        adaptive_escalation = metric("integrated_memory_v1_3_condition_adaptive", "escalation_rate")
        adaptive_slowdown = metric("integrated_memory_v1_3_condition_adaptive", "slowdown_vs_transformer")
        cond32_exact = metric("integrated_memory_v1_2_condition_v3_cond32", "exact_accuracy")
        cond32_speed = metric("integrated_memory_v1_2_condition_v3_cond32", "inference_examples_per_sec")
        adaptive_speed = metric("integrated_memory_v1_3_condition_adaptive", "inference_examples_per_sec")
        speedup_vs_32 = adaptive_speed / cond32_speed if adaptive_speed is not None and cond32_speed not in {None, 0.0} else None
        lines = [
            "## Integrated Memory v1.3 Adaptive Condition Candidate Budget",
            "",
            "This run tests inference-time escalation from condition K16 to condition K32 using non-oracle confidence signals.",
            "",
            f"Raw rows: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_integrated_memory.py --tasks {args.tasks} --models {args.models} --writer-mode {args.writer_mode} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k {args.candidate_k} --key-candidate-k {args.key_candidate_k} --condition-candidate-k {args.condition_candidate_k} --value-candidate-k {args.value_candidate_k} --template-split {args.template_split} --output-prefix {args.output_prefix}`",
            "",
            "### Final Budget Results",
            "",
            "| model | K | exact | condition recall | symbolic | learned | escalation | false escalation | missed escalation | failures avoided | train ex/sec | infer ex/sec | slowdown vs Transformer |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(diagnostic_rows, key=lambda item: str(item["model"])):
            lines.append(
                f"| {row['model']} | {row.get('candidate_k', '')} | {fmt(row.get('exact_accuracy_mean'))} | {fmt(row.get('condition_candidate_recall_mean'))} | {fmt(row.get('symbolic_reader_exact_mean'))} | {fmt(row.get('learned_slot_reader_exact_mean'))} | {fmt(row.get('escalation_rate_mean'))} | {fmt(row.get('false_escalation_rate_mean'))} | {fmt(row.get('missed_escalation_rate_mean'))} | {fmt(row.get('fraction_failures_avoided_mean'))} | {fmt(row.get('examples_per_sec_mean'))} | {fmt(row.get('inference_examples_per_sec_mean'))} | {fmt(row.get('slowdown_vs_transformer'))} |"
            )
        lines.extend(["", "### Verdict", ""])
        lines.append(f"adaptive exact = {fmt(adaptive_exact)}")
        lines.append(f"adaptive condition recall = {fmt(adaptive_recall)}")
        lines.append(f"adaptive escalation rate = {fmt(adaptive_escalation)}")
        lines.append(f"adaptive speedup vs always-condition32 = {fmt(speedup_vs_32)}x")
        lines.append(f"adaptive slowdown vs Transformer = {fmt(adaptive_slowdown)}x")
        if adaptive_exact is not None and adaptive_exact >= 0.94 and adaptive_recall is not None and adaptive_recall >= 0.98 and speedup_vs_32 is not None and speedup_vs_32 > 1.05 and adaptive_escalation is not None and adaptive_escalation < 0.9:
            lines.append("Adaptive preserves the condition32 gate while reducing cost, so v1.3 is acceptable for this stress setting.")
        else:
            lines.append("Adaptive does not satisfy all gates; keep always-condition32 as the accuracy fallback and do not replace v1.2 yet.")
        lines.append("Broad scaling remains blocked.")
        section = "\n".join(lines) + "\n"
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Integrated Memory v1.3 Adaptive Condition Candidate Budget\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section, encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section, encoding="utf-8")
        return
    if args.output_prefix == "conditional_candidate_v12":
        diagnostic_rows = [row for row in rows if row["task"] == "noisy_conditional" and row["template_split"] == "random"]

        def row_for(model: str) -> Dict[str, object] | None:
            for row in diagnostic_rows:
                if row["model"] == model:
                    return row
            return None

        def metric(model: str, name: str) -> float | None:
            row = row_for(model)
            if row is None:
                return None
            value = row.get(f"{name}_mean", row.get(name, ""))
            return None if value == "" else float(value)

        debug_path = Path(args.out_dir) / f"{args.output_prefix}_debug.jsonl"
        debug_counts: Dict[str, int] = {}
        if debug_path.exists():
            for line in debug_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                category = str(json.loads(line).get("failure_category", "unknown"))
                debug_counts[category] = debug_counts.get(category, 0) + 1
        debug_summary = ", ".join(f"{key}={value}" for key, value in sorted(debug_counts.items()))

        baseline_exact = metric("integrated_memory_v1_final", "exact_accuracy")
        k16_exact = metric("integrated_memory_v1_2_condition_v3_k16", "exact_accuracy")
        cond24_exact = metric("integrated_memory_v1_2_condition_v3_cond24", "exact_accuracy")
        cond32_exact = metric("integrated_memory_v1_2_condition_v3_cond32", "exact_accuracy")
        cond24_recall = metric("integrated_memory_v1_2_condition_v3_cond24", "condition_candidate_recall")
        cond24_gap = metric("integrated_memory_v1_2_condition_v3_cond24", "reader_symbolic_gap")
        cond24_slowdown = metric("integrated_memory_v1_2_condition_v3_cond24", "slowdown_vs_transformer")
        cond32_recall = metric("integrated_memory_v1_2_condition_v3_cond32", "condition_candidate_recall")
        cond32_gap = metric("integrated_memory_v1_2_condition_v3_cond32", "reader_symbolic_gap")
        cond32_slowdown = metric("integrated_memory_v1_2_condition_v3_cond32", "slowdown_vs_transformer")
        lines = [
            "## Integrated Memory v1.2 Conditional Candidate Recall Fix",
            "",
            "This run tests the conditional stress cell only, with asymmetric candidate budgets for condition fields while keeping key/value K fixed at 16.",
            "",
            f"Raw rows: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_integrated_memory.py --tasks {args.tasks} --models {args.models} --writer-mode {args.writer_mode} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k {args.candidate_k} --key-candidate-k {args.key_candidate_k} --condition-candidate-k {args.condition_candidate_k} --value-candidate-k {args.value_candidate_k} --template-split {args.template_split} --debug-examples {args.debug_examples} --output-prefix {args.output_prefix}`",
            "",
            "### Final Budget Results",
            "",
            "| model | candidate K | exact | condition recall | condition precision | condition pool | exact slot available | symbolic reader | learned reader | reader-symbolic gap | missing condition | slot F1 | ex/sec | slowdown |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(diagnostic_rows, key=lambda item: str(item["model"])):
            lines.append(
                f"| {row['model']} | {row.get('candidate_k', '')} | {fmt(row.get('exact_accuracy_mean'))} | {fmt(row.get('condition_candidate_recall_mean'))} | {fmt(row.get('condition_candidate_precision_mean'))} | {fmt(row.get('condition_pool_size_mean'))} | {fmt(row.get('exact_slot_available_rate_mean'))} | {fmt(row.get('symbolic_reader_exact_mean'))} | {fmt(row.get('learned_slot_reader_exact_mean'))} | {fmt(row.get('reader_symbolic_gap_mean'))} | {fmt(row.get('missing_condition_rate_mean'))} | {fmt(row.get('slot_f1_mean'))} | {fmt(row.get('examples_per_sec_mean'))} | {fmt(row.get('slowdown_vs_transformer'))} |"
            )
        lines.extend(["", "### Verdict", ""])
        lines.append(f"v1.1 baseline exact = {fmt(baseline_exact)}")
        lines.append(f"v1.2 k16 exact = {fmt(k16_exact)}")
        lines.append(f"v1.2 condition24 exact = {fmt(cond24_exact)}")
        if cond32_exact is not None:
            lines.append(f"v1.2 condition32 exact = {fmt(cond32_exact)}")
        lines.append(f"condition24 condition recall = {fmt(cond24_recall)}")
        if cond32_recall is not None:
            lines.append(f"condition32 condition recall = {fmt(cond32_recall)}")
        lines.append(f"condition24 reader-symbolic gap = {fmt(cond24_gap)}")
        if cond32_gap is not None:
            lines.append(f"condition32 reader-symbolic gap = {fmt(cond32_gap)}")
        lines.append(f"condition24 slowdown vs Transformer = {fmt(cond24_slowdown)}x")
        if cond32_slowdown is not None:
            lines.append(f"condition32 slowdown vs Transformer = {fmt(cond32_slowdown)}x")
        if debug_summary:
            lines.append(f"debug failure categories = {debug_summary}")
        if cond32_exact is not None:
            lines.append("Note: the condition32 row is from the optional fallback run with `--condition-candidate-k 32`; condition24 remains the requested minimal-cost diagnostic.")
        if cond24_exact is not None and cond24_exact >= 0.90 and cond24_recall is not None and cond24_recall > 0.8870:
            lines.append("Condition24 clears the random stress target and improves condition recall, so v1.2 should replace v1.1 for this conditional stress setting.")
        elif cond32_exact is not None and cond32_exact >= 0.90:
            lines.append("Condition24 did not clear the target, but condition32 did; condition32 is an expensive fallback rather than the default.")
        elif cond24_recall is not None and cond24_recall > 0.8870:
            lines.append("Condition recall improved but exact did not clear the target; the remaining issue is reader calibration/integration rather than candidate recall alone.")
        else:
            lines.append("Condition recall did not improve enough; inspect debug examples before adding another synthetic patch.")
        lines.append("Broad scaling remains blocked until the conditional stress cell is consistently above the gate.")
        section = "\n".join(lines) + "\n"
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Integrated Memory v1.2 Conditional Candidate Recall Fix\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section, encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section, encoding="utf-8")
        return
    if args.output_prefix == "conditional_stress_diagnostic":
        diagnostic_rows = [row for row in rows if row["task"] == "noisy_conditional"]

        def mean_for(model: str, split: str, metric: str) -> float | None:
            for row in diagnostic_rows:
                if row["model"] == model and row["template_split"] == split:
                    value = row.get(f"{metric}_mean", row.get(metric, ""))
                    return None if value == "" else float(value)
            return None

        debug_path = Path(args.out_dir) / f"{args.output_prefix}_debug.jsonl"
        debug_counts: Dict[str, int] = {}
        if debug_path.exists():
            for line in debug_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                category = str(json.loads(line).get("failure_category", "unknown"))
                debug_counts[category] = debug_counts.get(category, 0) + 1
        debug_summary = ", ".join(f"{key}={value}" for key, value in sorted(debug_counts.items()))

        random_integrated = mean_for("integrated_memory_v1_final", "random", "exact_accuracy")
        random_oracle_slots = mean_for("integrated_memory_v1_final_oracle_slots", "random", "exact_accuracy")
        random_symbolic = mean_for("integrated_memory_v1_final_learned_writer_symbolic_reader", "random", "exact_accuracy")
        random_oracle_cond = mean_for("integrated_memory_v1_final_oracle_condition_candidates", "random", "exact_accuracy")
        random_oracle_key = mean_for("integrated_memory_v1_final_oracle_key_candidates", "random", "exact_accuracy")
        random_oracle_value = mean_for("integrated_memory_v1_final_oracle_value_candidates", "random", "exact_accuracy")
        random_condition_recall = mean_for("integrated_memory_v1_final", "random", "condition_recall")
        random_tuple_error = mean_for("integrated_memory_v1_final", "random", "tuple_scoring_error_rate")
        transformer_ex_sec = [
            float(row["examples_per_sec_mean"])
            for row in diagnostic_rows
            if row["model"] == "transformer_baseline" and row.get("examples_per_sec_mean", "") != ""
        ]
        memory_ex_sec = [
            float(row["examples_per_sec_mean"])
            for row in diagnostic_rows
            if row["model"] == "integrated_memory_v1_final" and row.get("examples_per_sec_mean", "") != ""
        ]
        slowdown = (
            statistics.mean(transformer_ex_sec) / statistics.mean(memory_ex_sec)
            if transformer_ex_sec and memory_ex_sec
            else None
        )
        lines = [
            "## Conditional Stress Failure Diagnostic",
            "",
            "This diagnostic isolates the `noisy_conditional` stress failure at `seq_len=1024`, `distractors=32`, `slot_count=16`, and K=16.",
            "",
            f"Raw rows: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_integrated_memory.py --tasks {args.tasks} --models {args.models} --writer-mode {args.writer_mode} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k {args.candidate_k} --template-split {args.template_split} --debug-examples {args.debug_examples} --output-prefix {args.output_prefix}`",
            "",
            "### Final Budget Results",
            "",
            "| split | mode | exact | condition recall | key recall | value recall | all-field recall | missing key | missing condition | missing value | slot F1 | tuple error | learned reader | symbolic reader | oracle slot reader | ex/sec | slowdown |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(diagnostic_rows, key=lambda item: (str(item["template_split"]), str(item["model"]))):
            lines.append(
                f"| {row['template_split']} | {row['model']} | {fmt(row.get('exact_accuracy_mean'))} | {fmt(row.get('condition_recall_mean'))} | {fmt(row.get('key_recall_mean'))} | {fmt(row.get('value_recall_mean'))} | {fmt(row.get('all_field_candidate_recall_mean'))} | {fmt(row.get('missing_key_rate_mean'))} | {fmt(row.get('missing_condition_rate_mean'))} | {fmt(row.get('missing_value_rate_mean'))} | {fmt(row.get('slot_f1_mean'))} | {fmt(row.get('tuple_scoring_error_rate_mean'))} | {fmt(row.get('learned_slot_reader_exact_mean'))} | {fmt(row.get('symbolic_reader_exact_mean'))} | {fmt(row.get('oracle_slot_reader_exact_mean'))} | {fmt(row.get('examples_per_sec_mean'))} | {fmt(row.get('slowdown_vs_transformer'))} |"
            )
        lines.extend(["", "### Verdict", ""])
        lines.append(f"random integrated exact = {fmt(random_integrated)}")
        lines.append(f"random oracle slots exact = {fmt(random_oracle_slots)}")
        lines.append(f"random learned-writer symbolic exact = {fmt(random_symbolic)}")
        lines.append(f"random oracle condition candidates exact = {fmt(random_oracle_cond)}")
        lines.append(f"random oracle key candidates exact = {fmt(random_oracle_key)}")
        lines.append(f"random oracle value candidates exact = {fmt(random_oracle_value)}")
        lines.append(f"random condition recall = {fmt(random_condition_recall)}")
        lines.append(f"random tuple scoring error rate = {fmt(random_tuple_error)}")
        if debug_summary:
            lines.append(f"debug failure categories = {debug_summary}")
        if slowdown is not None:
            lines.append(f"memory slowdown vs Transformer = {fmt(slowdown)}x")
        if random_oracle_slots is not None and random_oracle_slots >= 0.99 and random_symbolic is not None and random_symbolic < 0.90:
            lines.append("Oracle slots solve the read path while learned-writer symbolic remains low, so the writer/candidate extraction path is the bottleneck.")
        elif random_symbolic is not None and random_integrated is not None and random_symbolic > random_integrated + 0.05:
            lines.append("Learned-writer symbolic reading is higher than integrated learned reading, so the learned reader/integration path contributes to the failure.")
        if random_oracle_cond is not None and random_integrated is not None and random_oracle_cond > random_integrated + 0.05:
            lines.append("Oracle condition candidates improve the result, so condition candidate recall is the first field bottleneck.")
        elif random_oracle_key is not None and random_integrated is not None and random_oracle_key > random_integrated + 0.05:
            lines.append("Oracle key candidates improve the result, so key candidate recall is the first field bottleneck.")
        elif random_oracle_value is not None and random_integrated is not None and random_oracle_value > random_integrated + 0.05:
            lines.append("Oracle value candidates improve the result, so value candidate recall is the first field bottleneck.")
        elif random_tuple_error is not None and random_tuple_error > 0.05:
            lines.append("Candidates are mostly present but tuple errors remain, so tuple binding/scoring is the bottleneck.")
        lines.append("Broad scaling is not allowed from this diagnostic.")
        section = "\n".join(lines) + "\n"
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Conditional Stress Failure Diagnostic\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section, encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section, encoding="utf-8")
        return
    if args.output_prefix in {"integrated_memory_v11_final", "integrated_memory_v11_stress"}:
        is_stress = args.output_prefix == "integrated_memory_v11_stress"
        title = "Integrated Memory v1.1 Tiny Stress Check" if is_stress else "Integrated Memory v1.1 Final Synthetic Check"
        final_rows = [row for row in rows if row["model"] in {"transformer_baseline", "integrated_memory_v1_final"}]
        integrated_rows = [row for row in final_rows if row["model"] == "integrated_memory_v1_final"]
        transformer_rows = [row for row in final_rows if row["model"] == "transformer_baseline"]
        integrated_exact = [float(row["exact_accuracy_mean"]) for row in integrated_rows if row.get("exact_accuracy_mean", "") != ""]
        mean_integrated_exact = statistics.mean(integrated_exact) if integrated_exact else None
        gains = [
            float(row["memory_model_gain_over_transformer"])
            for row in integrated_rows
            if row.get("memory_model_gain_over_transformer", "") != ""
        ]
        mean_gain = statistics.mean(gains) if gains else None
        worst_row = min(integrated_rows, key=lambda row: float(row["exact_accuracy_mean"])) if integrated_rows else None
        worst_exact = float(worst_row["exact_accuracy_mean"]) if worst_row is not None else None
        all_cells_pass = bool(integrated_exact) and all(value >= 0.90 for value in integrated_exact)
        all_cells_beat = bool(gains) and all(value > 0.0 for value in gains)
        memory_ex_sec = [
            float(row["examples_per_sec_mean"])
            for row in integrated_rows
            if row.get("examples_per_sec_mean", "") != ""
        ]
        transformer_ex_sec = [
            float(row["examples_per_sec_mean"])
            for row in transformer_rows
            if row.get("examples_per_sec_mean", "") != ""
        ]
        mean_memory_ex_sec = statistics.mean(memory_ex_sec) if memory_ex_sec else None
        mean_transformer_ex_sec = statistics.mean(transformer_ex_sec) if transformer_ex_sec else None
        slowdown = (
            mean_transformer_ex_sec / mean_memory_ex_sec
            if mean_memory_ex_sec and mean_transformer_ex_sec
            else None
        )
        description = (
            "This tiny stress check keeps the repaired final integrated path fixed and raises the setting to `seq_len=1024`, `distractors=32`, `slot_count=16`, and seeds `0,1`."
            if is_stress
            else "This compact check uses the repaired integrated path: v4.5 condition writer for `noisy_conditional` and v4.6 coexisting writer for `noisy_coexisting`, both at K=16 with hard noise and marker-rate 0.0."
        )
        lines = [
            f"## {title}",
            "",
            description,
            "",
            f"Raw rows: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_integrated_memory.py --tasks {args.tasks} --models {args.models} --writer-mode {args.writer_mode} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k {args.candidate_k} --template-split {args.template_split} --debug-examples {args.debug_examples} --output-prefix {args.output_prefix}`",
            "",
            "### Final Budget Results",
            "",
            "| task | split | model | exact | condition exact | set exact | set F1 | slot F1 | key recall | condition recall | value recall | missing key | missing condition | missing value | train time | ex/sec | params | gain vs Transformer |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(final_rows, key=lambda item: (str(item["task"]), str(item["template_split"]), str(item["model"]))):
            lines.append(
                f"| {row['task']} | {row['template_split']} | {row['model']} | {fmt(row.get('exact_accuracy_mean'))} | {fmt(row.get('condition_exact_mean'))} | {fmt(row.get('set_exact_mean'))} | {fmt(row.get('set_f1_mean'))} | {fmt(row.get('slot_f1_mean'))} | {fmt(row.get('key_recall_mean'))} | {fmt(row.get('condition_recall_mean'))} | {fmt(row.get('value_recall_mean'))} | {fmt(row.get('missing_key_rate_mean'))} | {fmt(row.get('missing_condition_rate_mean'))} | {fmt(row.get('missing_value_rate_mean'))} | {fmt(row.get('train_time_mean'))} | {fmt(row.get('examples_per_sec_mean'))} | {fmt(row.get('parameter_count_mean'))} | {fmt(row.get('memory_model_gain_over_transformer'))} |"
            )
        lines.extend(["", "### Verdict", ""])
        if all_cells_beat:
            lines.append("Integrated memory beats the Transformer baseline on all four task/split cells.")
        else:
            lines.append("Integrated memory does not beat the Transformer baseline on every cell.")
        lines.append(f"Mean integrated-memory exact = {fmt(mean_integrated_exact)}.")
        lines.append(f"Mean gain over Transformer = {fmt(mean_gain)}.")
        if all_cells_pass:
            lines.append("Every integrated-memory cell is >= 0.90, so this check is passed.")
        else:
            if is_stress and worst_exact is not None and worst_exact >= 0.80:
                lines.append("Worst cell is in the 0.80-0.90 range; diagnose that cell and do not scale.")
            else:
                lines.append("At least one integrated-memory cell is below 0.90; do not scale until that cell is debugged.")
        if worst_row is not None:
            lines.append(
                f"Worst integrated cell: `{worst_row['task']} / {worst_row['template_split']}` exact = {fmt(worst_row.get('exact_accuracy_mean'))}."
            )
        if slowdown is not None:
            lines.append(
                f"Mean throughput: Transformer {fmt(mean_transformer_ex_sec)} ex/sec vs integrated memory {fmt(mean_memory_ex_sec)} ex/sec, about {fmt(slowdown)}x slower by examples/sec."
            )
        if is_stress:
            if slowdown is not None and slowdown > 20.0:
                lines.append("Memory remains >20x slower, so the next phase must be speed/algorithm work, not scaling.")
            lines.append("Broad scaling is not allowed from this stress check.")
        elif all_cells_beat and all_cells_pass:
            lines.append("Allowed next step: one tiny stress check only (`seq_len=1024`, `distractors=32`, `slot_count=16`, seeds `0,1`).")
        else:
            lines.append("Next step is debugging, not stress or scale.")
        section = "\n".join(lines) + "\n"
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = f"\n## {title}\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section, encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section, encoding="utf-8")
        return
    if args.output_prefix in {"integrated_coexisting_diagnostic", "integrated_coexisting_v46"}:
        def mean_for(model: str, split: str, metric: str) -> float | None:
            for row in rows:
                if row["model"] == model and row["template_split"] == split:
                    value = row.get(f"{metric}_mean", row.get(metric, ""))
                    return None if value == "" else float(value)
            return None

        heldout_integrated = mean_for("integrated_memory_v1", "heldout", "exact_accuracy")
        heldout_v46 = mean_for("integrated_memory_v1_coexisting_v2", "heldout", "exact_accuracy")
        heldout_oracle_slots = mean_for("integrated_memory_v1_oracle_slots", "heldout", "exact_accuracy")
        heldout_oracle_learned = mean_for("integrated_memory_v1_oracle_writer_learned_reader", "heldout", "exact_accuracy")
        heldout_learned_symbolic = mean_for("integrated_memory_v1_learned_writer_symbolic_reader", "heldout", "exact_accuracy")
        debug_counts: Dict[str, int] = {}
        debug_path = Path(args.out_dir) / f"{args.output_prefix}_debug.jsonl"
        if debug_path.exists():
            for line in debug_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                category = str(json.loads(line).get("failure_category", "unknown"))
                debug_counts[category] = debug_counts.get(category, 0) + 1
        debug_summary = ", ".join(f"{key}={value}" for key, value in sorted(debug_counts.items()))
        lines = [
            "## Coexisting Writer v4.6 Heldout Fix" if args.output_prefix == "integrated_coexisting_v46" else "## Integrated Memory v1 Coexisting Heldout Diagnostic",
            "",
            "This diagnostic isolates the weak `noisy_coexisting / heldout` cell by comparing learned writer+learned reader, oracle slots, oracle writer+learned reader, and learned writer+symbolic reader.",
            "",
            f"Raw rows: `{raw_count}`. Summary rows: `{len(summary)}`.",
            "",
            "Command:",
            "",
            f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_integrated_memory.py --tasks {args.tasks} --models {args.models} --writer-mode {args.writer_mode} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k {args.candidate_k} --template-split {args.template_split} --debug-examples {args.debug_examples} --output-prefix {args.output_prefix}`",
            "",
            "### Final Budget Results",
            "",
            "| split | model/mode | exact | set P | set R | set F1 | writer slot F1 | key recall | value recall | missing key | missing value | all-slots exact | missed pos | extra FP | gain vs Transformer |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(rows, key=lambda item: (str(item["template_split"]), str(item["model"]))):
            lines.append(
                f"| {row['template_split']} | {row['model']} | {fmt(row.get('exact_accuracy_mean'))} | {fmt(row.get('set_precision_mean'))} | {fmt(row.get('set_recall_mean'))} | {fmt(row.get('set_f1_mean'))} | {fmt(row.get('slot_f1_mean'))} | {fmt(row.get('key_recall_mean'))} | {fmt(row.get('value_recall_mean'))} | {fmt(row.get('missing_key_rate_mean'))} | {fmt(row.get('missing_value_rate_mean'))} | {fmt(row.get('all_slots_exact_mean'))} | {fmt(row.get('missed_positive_rate_mean'))} | {fmt(row.get('extra_false_positive_rate_mean'))} | {fmt(row.get('memory_model_gain_over_transformer'))} |"
            )
        lines.extend(["", "### Verdict", ""])
        lines.append(f"heldout integrated exact = {fmt(heldout_integrated)}")
        if heldout_v46 is not None:
            lines.append(f"heldout coexisting v4.6 exact = {fmt(heldout_v46)}")
        lines.append(f"heldout oracle slots exact = {fmt(heldout_oracle_slots)}")
        lines.append(f"heldout oracle writer + learned reader exact = {fmt(heldout_oracle_learned)}")
        lines.append(f"heldout learned writer + symbolic reader exact = {fmt(heldout_learned_symbolic)}")
        if debug_summary:
            lines.append(f"debug failure categories = {debug_summary}")
        if heldout_oracle_learned is not None and heldout_oracle_learned >= 0.95:
            lines.append("Oracle slots fix the learned reader path, so the structured set reader is not the main bottleneck.")
        if heldout_v46 is not None:
            if heldout_v46 >= 0.8:
                lines.append("Coexisting v4.6 clears the requested heldout threshold, so the coexisting writer fix worked.")
            else:
                lines.append("Coexisting v4.6 does not clear the requested heldout threshold.")
        if heldout_learned_symbolic is not None and heldout_integrated is not None:
            if args.output_prefix == "integrated_coexisting_v46" and heldout_learned_symbolic >= 0.95:
                lines.append("Learned writer + symbolic reader is also high, confirming the v4.6 writer extracted the needed slots; the reader remains innocent.")
            elif heldout_learned_symbolic < 0.8:
                lines.append("Symbolic reader does not fix learned-writer output; it removes set-reader false positives but remains weak, so the writer/slot extraction path is responsible.")
            elif heldout_learned_symbolic > heldout_integrated + 0.15:
                lines.append("Symbolic reader repairs learned-writer output, so the set reader is responsible.")
        gate_value = heldout_v46 if heldout_v46 is not None else heldout_integrated
        if gate_value is not None and gate_value < 0.8:
            lines.append("The smallest fix should target coexisting heldout writer/template extraction before scaling.")
            if debug_counts:
                dominant = max(debug_counts.items(), key=lambda item: item[1])[0]
                lines.append(f"The dominant observed failure is `{dominant}`.")
            lines.append("We are not ready to scale.")
        else:
            lines.append("The weak cell is repaired enough for a follow-up scale check.")
        section = "\n".join(lines) + "\n"
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Coexisting Writer v4.6 Heldout Fix\n" if args.output_prefix == "integrated_coexisting_v46" else "\n## Integrated Memory v1 Coexisting Heldout Diagnostic\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section, encoding="utf-8")
        else:
            path.write_text("# HPM-Lite Results\n\n" + section, encoding="utf-8")
        return
    lines = [
        "## Integrated Memory Model v1",
        "",
        "This compares a small local causal Transformer answer baseline against a typed-memory pipeline: Writer v4.5 with K=16, full K^3 tuple scoring, typed memory slots, and a learned structured reader. The writer/reader path is trained with extraction/readout objectives; the Transformer baseline is trained with answer CE.",
        "",
        f"Raw rows: `{raw_count}`. Summary rows: `{len(summary)}`.",
        "",
        "Command:",
        "",
        f"- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_integrated_memory.py --tasks {args.tasks} --models {args.models} --writer-mode {args.writer_mode} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device} --noise-levels {args.noise_levels} --marker-rates {args.marker_rates} --distractor-counts {args.distractor_counts} --slot-counts {args.slot_counts} --max-slots {args.max_slots} --candidate-k {args.candidate_k} --template-split {args.template_split}`",
        "",
        "### Final Budget Results",
        "",
        "| task | split | model | exact | CE | writer exact | slot F1 | condition recall | train time | ex/sec | params | gain vs Transformer |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["task"]), str(item["template_split"]), str(item["model"]))):
        lines.append(
            f"| {row['task']} | {row['template_split']} | {row['model']} | {fmt(row.get('exact_accuracy_mean'))} | {fmt(row.get('answer_ce_mean'))} | {fmt(row.get('writer_exact_mean'))} | {fmt(row.get('slot_f1_mean'))} | {fmt(row.get('condition_recall_mean'))} | {fmt(row.get('train_time_mean'))} | {fmt(row.get('examples_per_sec_mean'))} | {fmt(row.get('parameter_count_mean'))} | {fmt(row.get('memory_model_gain_over_transformer'))} |"
        )
    gains = [
        float(row["memory_model_gain_over_transformer"])
        for row in rows
        if row["model"] == "integrated_memory_v1" and row.get("memory_model_gain_over_transformer", "") != ""
    ]
    mean_gain = statistics.mean(gains) if gains else None
    integrated_exact = [
        float(row["exact_accuracy_mean"])
        for row in rows
        if row["model"] == "integrated_memory_v1" and row.get("exact_accuracy_mean", "") != ""
    ]
    min_integrated_exact = min(integrated_exact) if integrated_exact else None
    lines.extend(
        [
            "",
            "### Verdict",
            "",
            f"Mean integrated-memory gain at budget {final_budget}: {fmt(mean_gain)}",
            f"Worst integrated-memory exact at budget {final_budget}: {fmt(min_integrated_exact)}",
        ]
    )
    if mean_gain is not None and mean_gain > 0.05:
        lines.append("Integrated memory clearly beats the Transformer baseline in this run.")
    elif mean_gain is not None and mean_gain >= -0.02:
        lines.append("Integrated memory ties the Transformer baseline; complexity is not justified yet unless inspectability or speed matters.")
    else:
        lines.append("Integrated memory loses to the Transformer baseline; stop scaling and inspect the integration.")
    if min_integrated_exact is not None and min_integrated_exact < 0.8:
        lines.append("The gain is not uniformly solved: at least one task/split remains weak, so scaling model size is premature.")
    else:
        lines.append("All requested task/split cells are strong enough to consider a small scale-up check.")
    section = "\n".join(lines) + "\n"
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## Integrated Memory Model v1\n"
        index = old.find(marker)
        if index >= 0:
            old = old[:index].rstrip()
        path.write_text(old + "\n\n" + section, encoding="utf-8")
    else:
        path.write_text("# HPM-Lite Results\n\n" + section, encoding="utf-8")


def update_writeup(path: Path, summary: List[Dict[str, object]], args: argparse.Namespace) -> None:
    if args.output_prefix.endswith("_smoke"):
        return

    final_budget = max(parse_int_list(args.budgets))
    if args.output_prefix == "conditional_adaptive_v13":
        rows = [
            row
            for row in summary
            if int(row["budget_steps"]) == final_budget
            and row["task"] == "noisy_conditional"
            and row["template_split"] == "random"
        ]

        def metric(model: str, name: str) -> float | None:
            for row in rows:
                if row["model"] == model:
                    value = row.get(f"{name}_mean", "")
                    return None if value == "" else float(value)
            return None

        cond32_speed = metric("integrated_memory_v1_2_condition_v3_cond32", "inference_examples_per_sec")
        adaptive_speed = metric("integrated_memory_v1_3_condition_adaptive", "inference_examples_per_sec")
        speedup_vs_32 = adaptive_speed / cond32_speed if adaptive_speed is not None and cond32_speed not in {None, 0.0} else None
        section = f"""## Integrated Memory v1.3 Adaptive Condition Candidate Budget

Adaptive v1.3 starts with condition K16 and escalates individual examples to condition K32 using non-oracle confidence signals: condition score margin, condition entropy, tuple margin, and reader margin.

| final budget | condition16 exact | condition24 exact | condition32 exact | adaptive exact | adaptive recall | escalation rate | speedup vs condition32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(metric("integrated_memory_v1_2_condition_v3_k16", "exact_accuracy"))} | {fmt(metric("integrated_memory_v1_2_condition_v3_cond24", "exact_accuracy"))} | {fmt(metric("integrated_memory_v1_2_condition_v3_cond32", "exact_accuracy"))} | {fmt(metric("integrated_memory_v1_3_condition_adaptive", "exact_accuracy"))} | {fmt(metric("integrated_memory_v1_3_condition_adaptive", "condition_candidate_recall"))} | {fmt(metric("integrated_memory_v1_3_condition_adaptive", "escalation_rate"))} | {fmt(speedup_vs_32)} |

Interpretation: keep adaptive only if it preserves condition32 accuracy and recall while being meaningfully faster and not escalating almost every example.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Integrated Memory v1.3 Adaptive Condition Candidate Budget\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return
    if args.output_prefix == "conditional_candidate_v12":
        rows = [
            row
            for row in summary
            if int(row["budget_steps"]) == final_budget
            and row["task"] == "noisy_conditional"
            and row["template_split"] == "random"
        ]

        def metric(model: str, name: str) -> float | None:
            for row in rows:
                if row["model"] == model:
                    value = row.get(f"{name}_mean", "")
                    return None if value == "" else float(value)
            return None

        section = f"""## Integrated Memory v1.2 Conditional Candidate Recall Fix

This fix keeps the v1.1 architecture fixed and only changes condition candidate recall: ConditionCandidateProposerV3 unions token, span, guideline, and local fact-window condition views, then allows an asymmetric condition budget.

| final budget | v1.1 exact | v1.2 K16 exact | condition24 exact | condition24 recall | condition32 exact | condition32 recall | reader-symbolic gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(metric("integrated_memory_v1_final", "exact_accuracy"))} | {fmt(metric("integrated_memory_v1_2_condition_v3_k16", "exact_accuracy"))} | {fmt(metric("integrated_memory_v1_2_condition_v3_cond24", "exact_accuracy"))} | {fmt(metric("integrated_memory_v1_2_condition_v3_cond24", "condition_candidate_recall"))} | {fmt(metric("integrated_memory_v1_2_condition_v3_cond32", "exact_accuracy"))} | {fmt(metric("integrated_memory_v1_2_condition_v3_cond32", "condition_candidate_recall"))} | {fmt(metric("integrated_memory_v1_2_condition_v3_cond32", "reader_symbolic_gap"))} |

Interpretation: condition24 is the minimal-cost repair but did not clear 0.90 in this run. Condition32 cleared the gate as an expensive fallback, so the bottleneck remains condition candidate recall/candidate budget rather than tuple binding or reader calibration. Broad scaling remains blocked.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Integrated Memory v1.2 Conditional Candidate Recall Fix\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return
    if args.output_prefix == "conditional_stress_diagnostic":
        rows = [
            row
            for row in summary
            if int(row["budget_steps"]) == final_budget
            and row["task"] == "noisy_conditional"
            and row["template_split"] == "random"
        ]

        def exact(model: str) -> float | None:
            for row in rows:
                if row["model"] == model:
                    value = row.get("exact_accuracy_mean", "")
                    return None if value == "" else float(value)
            return None

        def metric(model: str, name: str) -> float | None:
            for row in rows:
                if row["model"] == model:
                    value = row.get(f"{name}_mean", "")
                    return None if value == "" else float(value)
            return None

        section = f"""## Conditional Stress Failure Diagnostic

This diagnostic decomposes the remaining conditional stress failure into oracle slots, symbolic readout, and field-specific candidate oracles.

| final budget | integrated | oracle slots | learned writer + symbolic | oracle condition candidates | oracle key candidates | oracle value candidates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(exact("integrated_memory_v1_final"))} | {fmt(exact("integrated_memory_v1_final_oracle_slots"))} | {fmt(exact("integrated_memory_v1_final_learned_writer_symbolic_reader"))} | {fmt(exact("integrated_memory_v1_final_oracle_condition_candidates"))} | {fmt(exact("integrated_memory_v1_final_oracle_key_candidates"))} | {fmt(exact("integrated_memory_v1_final_oracle_value_candidates"))} |

Random-split condition recall for the learned writer is {fmt(metric("integrated_memory_v1_final", "condition_recall"))}; tuple scoring error rate is {fmt(metric("integrated_memory_v1_final", "tuple_scoring_error_rate"))}.

Interpretation: oracle slots test the reader, symbolic readout tests whether learned slots contain the right tuple, and field-specific oracle candidates identify which candidate pool fails first. Broad scaling remains blocked until this conditional stress cell is repaired.
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Conditional Stress Failure Diagnostic\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return
    if args.output_prefix in {"integrated_memory_v11_final", "integrated_memory_v11_stress"}:
        is_stress = args.output_prefix == "integrated_memory_v11_stress"
        title = "Integrated Memory v1.1 Tiny Stress Check" if is_stress else "Integrated Memory v1.1 Final Synthetic Check"
        rows = [
            row
            for row in summary
            if int(row["budget_steps"]) == final_budget and row["model"] == "integrated_memory_v1_final"
        ]
        exact_values = [float(row["exact_accuracy_mean"]) for row in rows if row.get("exact_accuracy_mean", "") != ""]
        gains = [float(row["memory_model_gain_over_transformer"]) for row in rows if row.get("memory_model_gain_over_transformer", "") != ""]
        worst_row = min(rows, key=lambda row: float(row["exact_accuracy_mean"])) if rows else None
        interpretation = (
            "Interpretation: this is a deliberately small harder setting. Passing it permits more diagnosis, but broad scaling remains blocked, especially if the memory path is still much slower than the Transformer baseline."
            if is_stress
            else "Interpretation: if all four cells beat the Transformer baseline and every integrated cell is at least 0.90 exact, the repaired synthetic suite is passed. The next step should be only a tiny stress check, not broad scaling."
        )
        section = f"""## {title}

{"The tiny stress check raises sequence length, distractors, and slot count while keeping the repaired integrated model fixed." if is_stress else "The final synthetic check combines the repaired condition and coexisting writers into one integrated model mode."}

| final budget | cells | mean exact | worst cell | worst exact | mean gain vs Transformer |
| ---: | ---: | ---: | --- | ---: | ---: |
| {final_budget} | {len(rows)} | {fmt(statistics.mean(exact_values) if exact_values else None)} | {f"{worst_row['task']} / {worst_row['template_split']}" if worst_row else ""} | {fmt(worst_row.get("exact_accuracy_mean") if worst_row else None)} | {fmt(statistics.mean(gains) if gains else None)} |

{interpretation}
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = f"\n## {title}\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return
    if args.output_prefix in {"integrated_coexisting_diagnostic", "integrated_coexisting_v46"}:
        rows = [row for row in summary if int(row["budget_steps"]) == final_budget and row["template_split"] == "heldout"]

        def exact(model: str) -> float | None:
            for row in rows:
                if row["model"] == model:
                    value = row.get("exact_accuracy_mean", "")
                    return None if value == "" else float(value)
            return None

        if args.output_prefix == "integrated_coexisting_v46":
            interpretation = (
                "Interpretation: oracle slots and oracle-writer learned-reader remain high, so the structured set reader is still verified. "
                "Coexisting v4.6 and learned-writer symbolic-reader are both high, so the key/value writer extraction failure is repaired in this heldout diagnostic."
            )
        else:
            interpretation = (
                "Interpretation: if oracle slots and oracle-writer learned-reader are high but learned-writer symbolic-reader stays low, "
                "the writer/slot extraction path is the bottleneck. If symbolic fixes learned slots, the set reader is the bottleneck."
            )

        section = f"""## {"Coexisting Writer v4.6 Heldout Fix" if args.output_prefix == "integrated_coexisting_v46" else "Integrated Coexisting Heldout Diagnostic"}

The coexisting heldout diagnostic decomposes the weak integrated cell into oracle slots, oracle writer plus learned reader, and learned writer plus symbolic reader.

| final budget | integrated | coexisting v4.6 | oracle slots | oracle writer + learned reader | learned writer + symbolic reader |
| ---: | ---: | ---: | ---: | ---: | ---: |
| {final_budget} | {fmt(exact("integrated_memory_v1"))} | {fmt(exact("integrated_memory_v1_coexisting_v2"))} | {fmt(exact("integrated_memory_v1_oracle_slots"))} | {fmt(exact("integrated_memory_v1_oracle_writer_learned_reader"))} | {fmt(exact("integrated_memory_v1_learned_writer_symbolic_reader"))} |

{interpretation}
"""
        if path.exists():
            old = path.read_text(encoding="utf-8").rstrip()
            marker = "\n## Coexisting Writer v4.6 Heldout Fix\n" if args.output_prefix == "integrated_coexisting_v46" else "\n## Integrated Coexisting Heldout Diagnostic\n"
            index = old.find(marker)
            if index >= 0:
                old = old[:index].rstrip()
            path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
        else:
            path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")
        return
    rows = [row for row in summary if int(row["budget_steps"]) == final_budget and row["model"] == "integrated_memory_v1"]
    gains = [float(row["memory_model_gain_over_transformer"]) for row in rows if row.get("memory_model_gain_over_transformer", "") != ""]
    mean_gain = statistics.mean(gains) if gains else None
    exact_values = [float(row["exact_accuracy_mean"]) for row in rows if row.get("exact_accuracy_mean", "") != ""]
    min_exact = min(exact_values) if exact_values else None
    section = f"""## Integrated Memory Model v1

The integrated comparison moves from isolated writer/reader diagnostics to a model-level baseline: a tiny local Transformer trained with answer CE versus Writer v4.5 plus typed slots and a learned structured reader.

| final budget | mean memory gain vs Transformer | worst integrated exact |
| ---: | ---: | ---: |
| {final_budget} | {fmt(mean_gain)} | {fmt(min_exact)} |

Interpretation: a clear positive gain means typed memory is useful as a model component on these synthetic noisy tasks, but a weak worst-case cell means the next step is integration/debugging rather than model scaling.
"""
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## Integrated Memory Model v1\n"
        index = old.find(marker)
        if index >= 0:
            old = old[:index].rstrip()
        path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
    else:
        path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Transformer-only and typed-memory integrated models.")
    parser.add_argument("--tasks", type=str, default="noisy_conditional,noisy_coexisting")
    parser.add_argument("--models", type=str, default="transformer_baseline,integrated_memory_v1")
    parser.add_argument("--writer-mode", choices=["frozen", "trainable"], default="frozen")
    parser.add_argument("--budgets", type=str, default="300,1000")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--reader-hidden", type=int, default=128)
    parser.add_argument("--extractor-hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--reader-pretrain-steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--noise-levels", type=str, default="hard")
    parser.add_argument("--marker-rates", type=str, default="0.0")
    parser.add_argument("--distractor-counts", type=str, default="16")
    parser.add_argument("--slot-counts", type=str, default="8")
    parser.add_argument("--max-slots", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--key-candidate-k", type=int, default=16)
    parser.add_argument("--condition-candidate-k", type=int, default=16)
    parser.add_argument("--value-candidate-k", type=int, default=16)
    parser.add_argument("--adaptive-use-condition-margin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-use-condition-top", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-use-condition-entropy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-use-tuple-margin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-use-reader-confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-condition-margin-threshold", type=float, default=0.15)
    parser.add_argument("--adaptive-condition-top-threshold", type=float, default=-5.0)
    parser.add_argument("--adaptive-condition-entropy-threshold", type=float, default=2.2)
    parser.add_argument("--adaptive-tuple-margin-threshold", type=float, default=0.75)
    parser.add_argument("--adaptive-reader-margin-threshold", type=float, default=0.5)
    parser.add_argument("--candidate-loss-weight", type=float, default=1.0)
    parser.add_argument("--tuple-loss-weight", type=float, default=1.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    parser.add_argument("--template-mix", type=str, default="mixed")
    parser.add_argument("--template-split", type=str, default="heldout,random")
    parser.add_argument("--template-augmentation", type=str, default="extreme")
    parser.add_argument("--simplified-aux-weight", type=float, default=0.5)
    parser.add_argument("--guideline-loss-weight", type=float, default=2.0)
    parser.add_argument("--debug-examples", type=int, default=0)
    parser.add_argument("--output-prefix", type=str, default="integrated_memory_v1")
    parser.add_argument("--out-dir", type=str, default="runs")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tasks = parse_str_list(args.tasks, TASKS)
    models = parse_str_list(args.models, MODELS)
    budgets = sorted(set(parse_int_list(args.budgets)))
    seeds = parse_int_list(args.seeds)
    template_splits = parse_str_list(args.template_split, TEMPLATE_SPLITS)
    noise_levels = parse_str_list(args.noise_levels)
    marker_rates = parse_float_list(args.marker_rates)
    distractor_counts = parse_int_list(args.distractor_counts)
    slot_counts = parse_int_list(args.slot_counts)
    if args.candidate_k != 16:
        raise ValueError("Integrated Memory v1 freezes candidate_k=16; non-default K is intentionally disabled.")
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    projected_rows = (
        len(tasks)
        * len(models)
        * len(budgets)
        * len(seeds)
        * len(template_splits)
        * len(noise_levels)
        * len(marker_rates)
        * len(distractor_counts)
        * len(slot_counts)
    )
    print(f"projected_rows={projected_rows}", flush=True)
    raw_rows: List[Dict[str, object]] = []
    debug_examples: List[Dict[str, object]] = []
    memory_models = [model for model in models if model.startswith("integrated_memory_v1")]
    raw_path = out_dir / f"{args.output_prefix}_raw.csv"
    summary_path = out_dir / f"{args.output_prefix}_summary.csv"
    debug_path = out_dir / f"{args.output_prefix}_debug.jsonl"
    for task in tasks:
        for template_split in template_splits:
            for noise_level in noise_levels:
                for marker_rate in marker_rates:
                    for distractor_count in distractor_counts:
                        for slot_count in slot_counts:
                            for seed in seeds:
                                if "transformer_baseline" in models:
                                    raw_rows.extend(
                                        train_transformer_config(
                                            task,
                                            seed,
                                            template_split,
                                            noise_level,
                                            marker_rate,
                                            distractor_count,
                                            slot_count,
                                            budgets,
                                            args,
                                            device,
                                        )
                                    )
                                if memory_models:
                                    raw_rows.extend(
                                        train_integrated_config(
                                            task,
                                            seed,
                                            template_split,
                                            noise_level,
                                            marker_rate,
                                            distractor_count,
                                            slot_count,
                                            budgets,
                                            memory_models,
                                            args,
                                            device,
                                            debug_examples,
                                        )
                                    )
                                write_csv(raw_path, raw_rows, RAW_COLUMNS)
                                summary = summarize(raw_rows)
                                write_csv(summary_path, summary, SUMMARY_COLUMNS)
                                print(
                                    f"done task={task} split={template_split} noise={noise_level} marker={marker_rate} "
                                    f"distract={distractor_count} slots={slot_count} seed={seed}",
                                    flush=True,
                                )
    summary = summarize(raw_rows)
    write_csv(raw_path, raw_rows, RAW_COLUMNS)
    write_csv(summary_path, summary, SUMMARY_COLUMNS)
    if args.debug_examples > 0:
        with debug_path.open("w", encoding="utf-8") as handle:
            for item in debug_examples[: args.debug_examples]:
                handle.write(json.dumps(item) + "\n")
    update_results(Path("results.md"), summary, args, len(raw_rows))
    update_writeup(Path("writeup_structured_memory_result.md"), summary, args)
    print(f"wrote {raw_path}, {summary_path}, results.md", flush=True)


if __name__ == "__main__":
    main()
