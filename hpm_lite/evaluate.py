from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import torch

from .data import CONDITIONAL_TASKS, FactRecallConfig, FactRecallDataset, NO_VALUE, VOCAB_SIZE
from .metrics import (
    answer_cross_entropy,
    answer_span_correct_mask,
    answer_span_exact_accuracy,
    count_parameters,
    retrieval_correct_mask,
    retrieval_metrics,
)
from .model import HpmLiteConfig, HpmLiteModel
from .utils import resolve_device, set_seed, str_to_bool
from .write_modes import apply_write_mode


@torch.no_grad()
def evaluate_batches(
    model: HpmLiteModel,
    dataset: FactRecallDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    task: str,
    top_k: int,
    memory_control: str = "normal",
    write_mode: str = "oracle",
) -> Dict[str, float]:
    model.eval()
    start_time = time.perf_counter()
    total_examples = 0
    ce_sum = 0.0
    acc_sum = 0.0
    ret_top1_sum = 0.0
    ret_topk_sum = 0.0
    ret_margin_sum = 0.0
    ret_count = 0
    reasoning_sum = 0.0
    reasoning_count = 0
    cond_positive_correct = 0.0
    cond_positive_total = 0
    cond_negative_correct = 0.0
    cond_negative_total = 0
    cond_no_value_predictions = 0
    cond_target_no_value = 0
    cond_total = 0
    writer_sums = {
        "avg_written_slots": 0.0,
        "true_fact_written_rate": 0.0,
        "false_write_rate": 0.0,
        "missed_fact_rate": 0.0,
    }
    writer_count = 0

    for _ in range(batches):
        batch = dataset.sample_batch(batch_size, device=device)
        batch, write_stats = apply_write_mode(batch, write_mode)
        output = model(
            batch["input_ids"],
            memory_token_positions=batch["memory_token_positions"],
            memory_mask=batch["memory_mask"],
            answer_positions=batch["answer_positions"],
            query_key_positions=batch["query_key_positions"],
            top_k=top_k,
            task=task,
            hop_positive_memory_indices=batch["hop_positive_memory_indices"],
            memory_control=memory_control,
        )
        logits = output["logits"]
        ce = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        acc = answer_span_exact_accuracy(logits, batch["target_ids"], batch["loss_mask"])
        exact_mask = answer_span_correct_mask(logits, batch["target_ids"], batch["loss_mask"])
        if task in CONDITIONAL_TASKS:
            batch_index = torch.arange(logits.size(0), device=logits.device)
            predictions = logits[batch_index, batch["answer_positions"]].argmax(dim=-1)
            targets = batch["answer_tokens"]
            positive = targets != NO_VALUE
            negative = targets == NO_VALUE
            if positive.any():
                cond_positive_correct += exact_mask[positive].float().sum().item()
                cond_positive_total += int(positive.sum().item())
            if negative.any():
                cond_negative_correct += exact_mask[negative].float().sum().item()
                cond_negative_total += int(negative.sum().item())
            cond_no_value_predictions += int((predictions == NO_VALUE).sum().item())
            cond_target_no_value += int(negative.sum().item())
            cond_total += int(targets.numel())

        total_examples += batch_size
        ce_sum += ce.item() * batch_size
        acc_sum += acc.item() * batch_size
        for key in writer_sums:
            writer_sums[key] += write_stats[key] * batch_size
        writer_count += batch_size

        ret = retrieval_metrics(
            output["retrieval"],
            positive_indices=batch["positive_memory_indices"],
            positive_mask=batch.get("positive_memory_mask"),
        )
        if ret:
            ret_top1_sum += ret["retrieval_top1"] * batch_size
            ret_topk_sum += ret["retrieval_topk"] * batch_size
            ret_margin_sum += ret.get("retrieval_margin", 0.0) * batch_size
            ret_count += batch_size
            correct_retrieval = retrieval_correct_mask(
                output["retrieval"],
                positive_indices=batch["positive_memory_indices"],
                positive_mask=batch.get("positive_memory_mask"),
            )
            if correct_retrieval is not None and correct_retrieval.any():
                reasoning_sum += exact_mask[correct_retrieval].float().sum().item()
                reasoning_count += int(correct_retrieval.sum().item())

    metrics = {
        "answer_ce": ce_sum / max(total_examples, 1),
        "answer_exact": acc_sum / max(total_examples, 1),
        "examples": float(total_examples),
        "examples_per_sec": total_examples / max(time.perf_counter() - start_time, 1.0e-9),
    }
    if ret_count:
        metrics.update(
            {
                "retrieval_top1": ret_top1_sum / ret_count,
                "retrieval_topk": ret_topk_sum / ret_count,
                "retrieval_margin": ret_margin_sum / ret_count,
            }
        )
    if reasoning_count:
        metrics["reasoning_success_given_retrieval"] = reasoning_sum / reasoning_count
    if writer_count:
        metrics.update({key: value / writer_count for key, value in writer_sums.items()})
    if task in CONDITIONAL_TASKS and cond_total:
        metrics.update(
            {
                "positive_condition_accuracy": cond_positive_correct / max(cond_positive_total, 1),
                "negative_condition_accuracy": cond_negative_correct / max(cond_negative_total, 1),
                "memory_required_accuracy": cond_positive_correct / max(cond_positive_total, 1),
                "condition_binding_exact": cond_positive_correct / max(cond_positive_total, 1),
                "no_value_bias_rate": cond_no_value_predictions / cond_total,
                "target_no_value_rate": cond_target_no_value / cond_total,
                "target_value_rate": (cond_total - cond_target_no_value) / cond_total,
                "predicted_value_rate": (cond_total - cond_no_value_predictions) / cond_total,
            }
        )
    return metrics


def parse_seq_lens(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a tiny HPM-Lite checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model", choices=["local", "recurrent", "epmem", "hpm_lite", "hebbian"], default="epmem")
    parser.add_argument(
        "--task",
        choices=[
            "kv",
            "twohop",
            "coexisting",
            "conditional",
            "conditional_balanced",
            "conditional_positive_only",
            "conditional_contrastive",
            "longhop",
        ],
        default="kv",
    )
    parser.add_argument("--seq-lens", type=str, default="256,512,1024")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--oracle-memory", type=str_to_bool, default=True)
    parser.add_argument(
        "--memory-control",
        choices=["normal", "shuffle_values", "shuffled_values", "random_keys", "corrupt_values", "no_retrieval"],
        default="normal",
    )
    parser.add_argument("--write-mode", choices=["oracle", "fact_token", "random_write"], default="oracle")
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument("--repeated-keys", type=str_to_bool, default=False)
    parser.add_argument("--similar-values", type=str_to_bool, default=False)
    parser.add_argument("--distractor-fact-spans", type=int, default=0)
    parser.add_argument("--query-key-noise-only", type=str_to_bool, default=False)
    parser.add_argument("--fact-order", choices=["random", "query_last"], default="random")
    return parser


def config_from_args(args: argparse.Namespace) -> HpmLiteConfig:
    return HpmLiteConfig(
        model_type=args.model,
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        window=args.window,
        max_seq_len=max(2048, max(parse_seq_lens(getattr(args, "seq_lens", "1024")))),
    )


def load_model(args: argparse.Namespace, device: torch.device) -> HpmLiteModel:
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        saved_config = checkpoint.get("model_config", {})
        config = HpmLiteConfig(**saved_config)
        model = HpmLiteModel(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        return model

    model = HpmLiteModel(config_from_args(args)).to(device)
    return model


def main(argv: Iterable[str] | None = None) -> Dict[str, Dict[str, float]]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model = load_model(args, device)

    results: Dict[str, Dict[str, float]] = {}
    for seq_len in parse_seq_lens(args.seq_lens):
        dataset = FactRecallDataset(
            FactRecallConfig(
                seq_len=seq_len,
                window=args.window,
                task=args.task,
                num_facts=args.num_facts,
                seed=args.seed + seq_len,
                oracle_memory=args.oracle_memory,
                repeated_keys=args.repeated_keys,
                similar_values=args.similar_values,
                distractor_fact_spans=args.distractor_fact_spans,
                query_key_noise_only=args.query_key_noise_only,
                fact_order=args.fact_order,
            )
        )
        metrics = evaluate_batches(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            batches=args.eval_batches,
            device=device,
            task=args.task,
            top_k=args.top_k,
            memory_control=args.memory_control,
            write_mode=args.write_mode,
        )
        metrics["parameters"] = float(count_parameters(model))
        results[str(seq_len)] = metrics

    print(json.dumps(results, indent=2, sort_keys=True))
    return results


if __name__ == "__main__":
    main()
