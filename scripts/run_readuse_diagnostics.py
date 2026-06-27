from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpm_lite.data import CONDITIONAL_TASKS, FactRecallConfig, FactRecallDataset, QUERY
from hpm_lite.evaluate import evaluate_batches
from hpm_lite.metrics import answer_cross_entropy
from hpm_lite.structured_readout import (
    CandidateSetReadout,
    structured_bce_loss,
    structured_set_metrics,
    symbolic_condition_binding_metrics,
)
from hpm_lite.train import TinyAdamW, make_model
from hpm_lite.utils import ensure_dir, resolve_device, set_seed
from hpm_lite.write_modes import apply_write_mode, first_positions, parse_write_modes


TASKS = ["coexisting", "conditional", "conditional_balanced", "conditional_positive_only", "conditional_contrastive"]
MODELS = ["local", "epmem", "hpm_lite", "hebbian"]
CONTROLS = ["normal", "no_retrieval", "shuffled_values", "random_keys"]


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str, allowed: set[str]) -> List[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"unknown value(s): {unknown}")
    return items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose MEMFAIL-Lite read/use failures.")
    parser.add_argument("--tasks", type=str, default="coexisting,conditional")
    parser.add_argument("--models", type=str, default="local,epmem,hpm_lite,hebbian")
    parser.add_argument("--write-modes", type=str, default="oracle,fact_token")
    parser.add_argument("--controls", type=str, default="normal,no_retrieval,shuffled_values,random_keys")
    parser.add_argument("--budgets", type=str, default="30,100,300")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--readout-lr", type=float, default=1.0e-2)
    parser.add_argument("--lambda-ret", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="runs")
    return parser


def dataset_for(task: str, seq_len: int, window: int, seed: int, num_facts: int) -> FactRecallDataset:
    return FactRecallDataset(
        FactRecallConfig(
            seq_len=seq_len,
            window=window,
            task=task,
            num_facts=num_facts,
            seed=seed,
            oracle_memory=True,
        )
    )


def verify_no_answer_in_memory(
    task: str,
    seq_len: int,
    window: int,
    seed: int,
    batch_size: int,
    num_facts: int,
    write_modes: List[str],
) -> int:
    dataset = dataset_for(task, seq_len, window, seed, num_facts)
    batch = dataset.sample_batch(batch_size)
    query_positions = first_positions(batch["input_ids"], QUERY)
    answer_input_positions = batch["answer_target_positions"] + 1
    for write_mode in write_modes:
        written, _ = apply_write_mode(batch, write_mode)
        for b in range(batch_size):
            valid_positions = written["memory_token_positions"][b, written["memory_mask"][b]]
            if valid_positions.numel() == 0:
                continue
            if torch.any(valid_positions >= query_positions[b]):
                raise AssertionError("memory write reached QUERY or a later token")
            if torch.any(torch.isin(valid_positions.reshape(-1), answer_input_positions[b])):
                raise AssertionError("future answer token leaked into memory writes")
    return batch_size


@torch.no_grad()
def evaluate_structured_batches(
    model: torch.nn.Module,
    readout: CandidateSetReadout,
    dataset: FactRecallDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    task: str,
    top_k: int,
    memory_control: str,
    write_mode: str,
) -> Dict[str, float]:
    model.eval()
    readout.eval()
    total = 0
    exact_sum = 0.0
    f1_sum = 0.0
    bce_sum = 0.0
    for _ in range(batches):
        batch = dataset.sample_batch(batch_size, device=device)
        batch, _ = apply_write_mode(batch, write_mode)
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
        metrics = structured_set_metrics(readout, batch, output["retrieval"], memory_control)
        exact_sum += metrics["structured_set_exact"] * batch_size
        f1_sum += metrics["structured_per_value_f1"] * batch_size
        bce_sum += metrics["structured_bce"] * batch_size
        total += batch_size
    return {
        "structured_set_exact": exact_sum / max(total, 1),
        "structured_per_value_f1": f1_sum / max(total, 1),
        "structured_bce": bce_sum / max(total, 1),
    }


@torch.no_grad()
def evaluate_symbolic_condition_batches(
    dataset: FactRecallDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    memory_control: str,
    write_mode: str,
) -> Dict[str, float]:
    if memory_control == "no_retrieval":
        return {"symbolic_readout_available": 0.0}

    sums: Dict[str, float] = {}
    total = 0
    for _ in range(batches):
        batch = dataset.sample_batch(batch_size, device=device)
        batch, _ = apply_write_mode(batch, write_mode)
        metrics = symbolic_condition_binding_metrics(batch, memory_control)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * batch_size
        total += batch_size
    return {key: value / max(total, 1) for key, value in sums.items()}


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


RAW_COLUMNS = [
    "task",
    "write_mode",
    "model",
    "seed",
    "budget_steps",
    "memory_control",
    "answer_exact",
    "answer_ce",
    "retrieval_top1",
    "retrieval_topk",
    "reasoning_success_given_retrieval",
    "positive_condition_accuracy",
    "negative_condition_accuracy",
    "memory_required_accuracy",
    "condition_binding_exact",
    "no_value_bias_rate",
    "target_no_value_rate",
    "target_value_rate",
    "predicted_value_rate",
    "answer_distribution",
    "structured_set_exact",
    "structured_per_value_f1",
    "structured_bce",
    "symbolic_readout_available",
    "condition_symbolic_exact",
    "condition_symbolic_slot_accuracy",
    "condition_symbolic_value_accuracy",
    "exact_match_available_rate",
    "ambiguous_exact_match_rate",
    "symbolic_binding_hit_1_rate",
    "ce_vs_symbolic_gap",
    "retrieval_vs_symbolic_gap",
    "examples_per_sec",
    "train_wall_time_sec",
    "true_fact_written_rate",
    "false_write_rate",
    "missed_fact_rate",
]

SUMMARY_COLUMNS = [
    "task",
    "write_mode",
    "model",
    "budget_steps",
    "memory_control",
    "n",
    "answer_exact_mean",
    "answer_exact_std",
    "answer_ce_mean",
    "answer_ce_std",
    "retrieval_top1_mean",
    "retrieval_top1_std",
    "retrieval_topk_mean",
    "retrieval_topk_std",
    "reasoning_success_given_retrieval_mean",
    "reasoning_success_given_retrieval_std",
    "positive_condition_accuracy_mean",
    "positive_condition_accuracy_std",
    "negative_condition_accuracy_mean",
    "negative_condition_accuracy_std",
    "memory_required_accuracy_mean",
    "memory_required_accuracy_std",
    "condition_binding_exact_mean",
    "condition_binding_exact_std",
    "no_value_bias_rate_mean",
    "no_value_bias_rate_std",
    "target_no_value_rate_mean",
    "target_no_value_rate_std",
    "target_value_rate_mean",
    "target_value_rate_std",
    "predicted_value_rate_mean",
    "predicted_value_rate_std",
    "structured_set_exact_mean",
    "structured_set_exact_std",
    "structured_per_value_f1_mean",
    "structured_per_value_f1_std",
    "structured_bce_mean",
    "structured_bce_std",
    "symbolic_readout_available_mean",
    "symbolic_readout_available_std",
    "condition_symbolic_exact_mean",
    "condition_symbolic_exact_std",
    "condition_symbolic_slot_accuracy_mean",
    "condition_symbolic_slot_accuracy_std",
    "condition_symbolic_value_accuracy_mean",
    "condition_symbolic_value_accuracy_std",
    "exact_match_available_rate_mean",
    "exact_match_available_rate_std",
    "ambiguous_exact_match_rate_mean",
    "ambiguous_exact_match_rate_std",
    "symbolic_binding_hit_1_rate_mean",
    "symbolic_binding_hit_1_rate_std",
    "ce_vs_symbolic_gap_mean",
    "ce_vs_symbolic_gap_std",
    "retrieval_vs_symbolic_gap_mean",
    "retrieval_vs_symbolic_gap_std",
    "examples_per_sec_mean",
    "examples_per_sec_std",
    "train_wall_time_sec_mean",
    "train_wall_time_sec_std",
    "true_fact_written_rate_mean",
    "true_fact_written_rate_std",
    "false_write_rate_mean",
    "false_write_rate_std",
    "missed_fact_rate_mean",
    "missed_fact_rate_std",
]


def summarize(raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str, int, str], List[Dict[str, object]]] = {}
    for row in raw_rows:
        key = (
            str(row["task"]),
            str(row["write_mode"]),
            str(row["model"]),
            int(row["budget_steps"]),
            str(row["memory_control"]),
        )
        grouped.setdefault(key, []).append(row)

    metrics = [column[:-5] for column in SUMMARY_COLUMNS if column.endswith("_mean")]
    rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        task, write_mode, model, budget, control = key
        group = grouped[key]
        out: Dict[str, object] = {
            "task": task,
            "write_mode": write_mode,
            "model": model,
            "budget_steps": budget,
            "memory_control": control,
            "n": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric, "") != ""]
            if values:
                out[f"{metric}_mean"] = statistics.mean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
        rows.append(out)
    return rows


def flush_outputs(out_dir: Path, raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows = summarize(raw_rows)
    write_csv(out_dir / "readuse_raw.csv", raw_rows, RAW_COLUMNS)
    write_csv(out_dir / "readuse_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(out_dir / "condition_binding_raw.csv", raw_rows, RAW_COLUMNS)
    write_csv(out_dir / "condition_binding_summary.csv", summary_rows, SUMMARY_COLUMNS)
    return summary_rows


def train_and_evaluate(
    task: str,
    model_name: str,
    write_mode: str,
    seed: int,
    budgets: List[int],
    controls: List[str],
    args: argparse.Namespace,
    device: torch.device,
    raw_rows: List[Dict[str, object]],
    out_dir: Path,
) -> None:
    set_seed(seed)
    model_args = SimpleNamespace(
        model=model_name,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        window=args.window,
        seq_len=args.seq_len,
    )
    model = make_model(model_args, device)
    readout = CandidateSetReadout().to(device)
    model_optimizer = TinyAdamW(model.parameters(), lr=args.lr)
    readout_optimizer = TinyAdamW(readout.parameters(), lr=args.readout_lr)
    train_dataset = dataset_for(task, args.seq_len, args.window, seed, args.num_facts)
    budget_set = set(budgets)
    max_budget = max(budgets)
    start = time.perf_counter()

    for step in range(1, max_budget + 1):
        model.train()
        readout.train()
        batch = train_dataset.sample_batch(args.batch_size, device=device)
        batch, _ = apply_write_mode(batch, write_mode)
        output = model(
            batch["input_ids"],
            memory_token_positions=batch["memory_token_positions"],
            memory_mask=batch["memory_mask"],
            answer_positions=batch["answer_positions"],
            query_key_positions=batch["query_key_positions"],
            top_k=args.top_k,
            task=task,
            hop_positive_memory_indices=batch["hop_positive_memory_indices"],
            memory_control="normal",
        )
        logits = output["logits"]
        answer_loss = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        retrieval_loss = output["retrieval"].get("retrieval_loss", logits.new_zeros(()))
        loss = answer_loss + args.lambda_ret * retrieval_loss
        if task == "coexisting":
            loss = loss + structured_bce_loss(readout, batch, output["retrieval"], memory_control="normal")
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for {task}/{model_name}/{write_mode}/seed{seed}/step{step}")

        model_optimizer.zero_grad(set_to_none=True)
        readout_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        model_optimizer.step()
        readout_optimizer.step()

        if step not in budget_set:
            continue

        train_wall = time.perf_counter() - start
        for control in controls:
            eval_dataset = dataset_for(task, args.seq_len, args.window, 200_000 + seed + step, args.num_facts)
            metrics = evaluate_batches(
                model=model,
                dataset=eval_dataset,
                batch_size=args.batch_size,
                batches=args.eval_batches,
                device=device,
                task=task,
                top_k=args.top_k,
                memory_control=control,
                write_mode=write_mode,
            )
            structured = {}
            symbolic = {}
            if task == "coexisting":
                structured = evaluate_structured_batches(
                    model=model,
                    readout=readout,
                    dataset=eval_dataset,
                    batch_size=args.batch_size,
                    batches=args.eval_batches,
                    device=device,
                    task=task,
                    top_k=args.top_k,
                    memory_control=control,
                    write_mode=write_mode,
                )
            if task in CONDITIONAL_TASKS:
                symbolic = evaluate_symbolic_condition_batches(
                    dataset=eval_dataset,
                    batch_size=args.batch_size,
                    batches=args.eval_batches,
                    device=device,
                    memory_control=control,
                    write_mode=write_mode,
                )
                if "condition_symbolic_exact" in symbolic:
                    symbolic["ce_vs_symbolic_gap"] = symbolic["condition_symbolic_exact"] - metrics["answer_exact"]
                    if "reasoning_success_given_retrieval" in metrics:
                        symbolic["retrieval_vs_symbolic_gap"] = (
                            symbolic["condition_symbolic_exact"] - metrics["reasoning_success_given_retrieval"]
                        )
                    elif "retrieval_topk" in metrics:
                        symbolic["retrieval_vs_symbolic_gap"] = (
                            symbolic["condition_symbolic_exact"] - metrics["retrieval_topk"]
                        )
            answer_distribution = ""
            if task in CONDITIONAL_TASKS:
                answer_distribution = (
                    f"target_no_value={metrics.get('target_no_value_rate', '')};"
                    f"pred_no_value={metrics.get('no_value_bias_rate', '')};"
                    f"target_value={metrics.get('target_value_rate', '')};"
                    f"pred_value={metrics.get('predicted_value_rate', '')}"
                )
            raw_rows.append(
                {
                    "task": task,
                    "write_mode": write_mode,
                    "model": model_name,
                    "seed": seed,
                    "budget_steps": step,
                    "memory_control": control,
                    "answer_exact": metrics["answer_exact"],
                    "answer_ce": metrics["answer_ce"],
                    "retrieval_top1": metrics.get("retrieval_top1", ""),
                    "retrieval_topk": metrics.get("retrieval_topk", ""),
                    "reasoning_success_given_retrieval": metrics.get("reasoning_success_given_retrieval", ""),
                    "positive_condition_accuracy": metrics.get("positive_condition_accuracy", ""),
                    "negative_condition_accuracy": metrics.get("negative_condition_accuracy", ""),
                    "memory_required_accuracy": metrics.get("memory_required_accuracy", ""),
                    "condition_binding_exact": metrics.get("condition_binding_exact", ""),
                    "no_value_bias_rate": metrics.get("no_value_bias_rate", ""),
                    "target_no_value_rate": metrics.get("target_no_value_rate", ""),
                    "target_value_rate": metrics.get("target_value_rate", ""),
                    "predicted_value_rate": metrics.get("predicted_value_rate", ""),
                    "answer_distribution": answer_distribution,
                    "structured_set_exact": structured.get("structured_set_exact", ""),
                    "structured_per_value_f1": structured.get("structured_per_value_f1", ""),
                    "structured_bce": structured.get("structured_bce", ""),
                    "symbolic_readout_available": symbolic.get("symbolic_readout_available", ""),
                    "condition_symbolic_exact": symbolic.get("condition_symbolic_exact", ""),
                    "condition_symbolic_slot_accuracy": symbolic.get("condition_symbolic_slot_accuracy", ""),
                    "condition_symbolic_value_accuracy": symbolic.get("condition_symbolic_value_accuracy", ""),
                    "exact_match_available_rate": symbolic.get("exact_match_available_rate", ""),
                    "ambiguous_exact_match_rate": symbolic.get("ambiguous_exact_match_rate", ""),
                    "symbolic_binding_hit_1_rate": symbolic.get("symbolic_binding_hit_1_rate", ""),
                    "ce_vs_symbolic_gap": symbolic.get("ce_vs_symbolic_gap", ""),
                    "retrieval_vs_symbolic_gap": symbolic.get("retrieval_vs_symbolic_gap", ""),
                    "examples_per_sec": metrics["examples_per_sec"],
                    "train_wall_time_sec": train_wall,
                    "true_fact_written_rate": metrics["true_fact_written_rate"],
                    "false_write_rate": metrics["false_write_rate"],
                    "missed_fact_rate": metrics["missed_fact_rate"],
                }
            )
        flush_outputs(out_dir, raw_rows)
        print(f"done task={task} model={model_name} write={write_mode} seed={seed} budget={step}", flush=True)


def value_for(
    rows: List[Dict[str, object]],
    task: str,
    model: str,
    budget: int,
    metric: str,
    write_mode: str = "fact_token",
    control: str = "normal",
) -> float | None:
    values = [
        float(row[f"{metric}_mean"])
        for row in rows
        if row["task"] == task
        and row["model"] == model
        and int(row["budget_steps"]) == budget
        and row["write_mode"] == write_mode
        and row["memory_control"] == control
        and row[f"{metric}_mean"] != ""
    ]
    return statistics.mean(values) if values else None


def update_results_md(path: Path, summary_rows: List[Dict[str, object]], args: argparse.Namespace, leak_checks: int) -> None:
    budgets = parse_int_list(args.budgets)
    final_budget = max(budgets)
    tasks = parse_str_list(args.tasks, set(TASKS))
    conditional_tasks = [task for task in tasks if task in CONDITIONAL_TASKS]
    lines = [
        "## Condition Binding Structured Readout",
        "",
        "This run tests whether a symbolic condition-binding operator can use the stored key-condition-value slots even when next-token CE decoding stays weak. It does not add a learned writer, larger model, JEPA, ANN, graph memory, GKA, Priming, RL, or extra HPM-Lite machinery.",
        "",
        f"Tasks: `{args.tasks}`. Models: `{args.models}`. Write modes: `{args.write_modes}`. Controls: `{args.controls}`.",
        f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Seq len/window: `{args.seq_len}` / `{args.window}`. Top-k: `{args.top_k}`.",
        f"Device request: `{args.device}`. Platform: `{platform.platform()}`. Torch: `{torch.__version__}`.",
        f"Leak checks passed on `{leak_checks}` generated examples.",
        "",
        "Command:",
        "",
        f"- `python scripts/run_readuse_diagnostics.py --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device}`",
        "",
        "Raw and summarized outputs: `runs/condition_binding_raw.csv`, `runs/condition_binding_summary.csv`.",
        "",
        "### Coexisting Structured Baseline",
        "",
        "Coexisting is readout-solved under structured set decoding when retrieval succeeds; this baseline is retained as a diagnostic control.",
        "",
        "| model | budget | CE exact | retrieval top-k | structured set exact | per-value F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for budget in budgets:
            exact = value_for(summary_rows, "coexisting", model, budget, "answer_exact") if "coexisting" in tasks else None
            ret = value_for(summary_rows, "coexisting", model, budget, "retrieval_topk") if "coexisting" in tasks else None
            structured = value_for(summary_rows, "coexisting", model, budget, "structured_set_exact") if "coexisting" in tasks else None
            f1 = value_for(summary_rows, "coexisting", model, budget, "structured_per_value_f1") if "coexisting" in tasks else None
            lines.append(
                f"| {model} | {budget} | {'' if exact is None else f'{exact:.4f}'} | {'' if ret is None else f'{ret:.4f}'} | {'' if structured is None else f'{structured:.4f}'} | {'' if f1 is None else f'{f1:.4f}'} |"
            )

    lines.extend(
        [
            "",
            "### Conditional Variant CE Metrics",
            "",
            "| task | model | budget | exact | positive exact | negative exact | binding exact | no-value pred | value pred | retrieval top-k | use if retrieved |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task in conditional_tasks:
        for model in MODELS:
            for budget in budgets:
                exact = value_for(summary_rows, task, model, budget, "answer_exact")
                pos = value_for(summary_rows, task, model, budget, "positive_condition_accuracy")
                neg = value_for(summary_rows, task, model, budget, "negative_condition_accuracy")
                binding = value_for(summary_rows, task, model, budget, "condition_binding_exact")
                bias = value_for(summary_rows, task, model, budget, "no_value_bias_rate")
                pred_value = value_for(summary_rows, task, model, budget, "predicted_value_rate")
                ret = value_for(summary_rows, task, model, budget, "retrieval_topk")
                use = value_for(summary_rows, task, model, budget, "reasoning_success_given_retrieval")
                lines.append(
                    f"| {task} | {model} | {budget} | {'' if exact is None else f'{exact:.4f}'} | {'' if pos is None else f'{pos:.4f}'} | {'' if neg is None else f'{neg:.4f}'} | {'' if binding is None else f'{binding:.4f}'} | {'' if bias is None else f'{bias:.4f}'} | {'' if pred_value is None else f'{pred_value:.4f}'} | {'' if ret is None else f'{ret:.4f}'} | {'' if use is None else f'{use:.4f}'} |"
                )

    co_ce = value_for(summary_rows, "coexisting", "epmem", final_budget, "answer_exact") if "coexisting" in tasks else None
    co_struct = value_for(summary_rows, "coexisting", "epmem", final_budget, "structured_set_exact") if "coexisting" in tasks else None
    pos_local = value_for(summary_rows, "conditional_positive_only", "local", final_budget, "answer_exact")
    pos_no_mem = value_for(summary_rows, "conditional_positive_only", "epmem", final_budget, "answer_exact", control="no_retrieval")
    pos_epmem = value_for(summary_rows, "conditional_positive_only", "epmem", final_budget, "answer_exact")
    pos_ret = value_for(summary_rows, "conditional_positive_only", "epmem", final_budget, "retrieval_topk")
    contrast_local = value_for(summary_rows, "conditional_contrastive", "local", final_budget, "answer_exact")
    contrast_no_mem = value_for(summary_rows, "conditional_contrastive", "epmem", final_budget, "answer_exact", control="no_retrieval")
    contrast_epmem = value_for(summary_rows, "conditional_contrastive", "epmem", final_budget, "answer_exact")
    contrast_ret = value_for(summary_rows, "conditional_contrastive", "epmem", final_budget, "retrieval_topk")
    hpm_values = []
    for task in conditional_tasks:
        hpm = value_for(summary_rows, task, "hpm_lite", final_budget, "answer_exact")
        epmem = value_for(summary_rows, task, "epmem", final_budget, "answer_exact")
        if hpm is not None and epmem is not None:
            hpm_values.append((task, hpm - epmem))
    hebb_contrast = value_for(summary_rows, "conditional_contrastive", "hebbian", final_budget, "answer_exact")
    lines.extend(["", "### Conditional Repair Verdict", ""])
    if co_ce is not None and co_struct is not None:
        if co_struct - co_ce >= 0.5:
            lines.append(f"- Structured readout fixes coexisting for epmem at {final_budget} steps ({co_ce:.3f} CE exact vs {co_struct:.3f} set exact), so the failure is decoder/composition, not memory retrieval.")
        else:
            lines.append(f"- Structured readout does not fix coexisting for epmem at {final_budget} steps ({co_ce:.3f} CE exact vs {co_struct:.3f} set exact); readout or retrieval remains suspect.")
    if pos_local is not None and pos_no_mem is not None and pos_epmem is not None:
        if max(pos_local, pos_no_mem) > 0.3:
            lines.append(f"- Positive-only conditional still has shortcut risk: local {pos_local:.3f}, epmem no-retrieval {pos_no_mem:.3f}.")
        elif pos_ret is not None and pos_ret >= 0.8 and pos_epmem < 0.5:
            lines.append(f"- Positive-only conditional has high retrieval ({pos_ret:.3f}) but low exact ({pos_epmem:.3f}); mark as qualifier-use failure.")
        else:
            lines.append(f"- Positive-only conditional: local {pos_local:.3f}, no-retrieval {pos_no_mem:.3f}, epmem normal {pos_epmem:.3f}.")
    if contrast_local is not None and contrast_no_mem is not None and contrast_epmem is not None:
        requires_memory = max(contrast_local, contrast_no_mem) < 0.3 and contrast_epmem > max(contrast_local, contrast_no_mem) + 0.2
        if requires_memory:
            lines.append(f"- Contrastive conditional requires memory in this run: local {contrast_local:.3f}, no-retrieval {contrast_no_mem:.3f}, epmem {contrast_epmem:.3f}.")
        elif contrast_ret is not None and contrast_ret >= 0.8 and contrast_epmem < 0.5:
            lines.append(f"- Contrastive conditional retrieves the matching slot ({contrast_ret:.3f}) but exact is low ({contrast_epmem:.3f}); mark as condition-binding/use failure.")
        else:
            lines.append(f"- Contrastive conditional remains ambiguous: local {contrast_local:.3f}, no-retrieval {contrast_no_mem:.3f}, epmem {contrast_epmem:.3f}.")
    if hpm_values:
        strongest = max(hpm_values, key=lambda item: item[1])
        lines.append(f"- HPM-Lite vs epmem: strongest final-budget repaired-conditional delta is {strongest[1] * 100:.1f} points on `{strongest[0]}`; recurrence remains unjustified if this is below 5-10 points.")
    if hebb_contrast is not None:
        lines.append(f"- Hebbian contrastive condition-binding exact at final budget is {hebb_contrast:.3f}; low performance means simple association is not enough for qualifier binding.")
    lines.extend(
        [
            "",
            "### Symbolic Binding Final-Budget Controls",
            "",
            "Local rows here are `local + external symbolic readout`, not pure local model behavior. `no_retrieval` disables symbolic memory readout and is reported as unavailable.",
            "",
            "| task | write | model | control | CE exact | symbolic exact | symbolic_binding_hit_1_rate | exact available | ambiguous exact | slot acc | value acc | CE-symbolic gap | retrieval-symbolic gap |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    write_modes = parse_write_modes(args.write_modes)
    for task in conditional_tasks:
        for write_mode in write_modes:
            for model in MODELS:
                for control in CONTROLS:
                    ce = value_for(summary_rows, task, model, final_budget, "answer_exact", write_mode=write_mode, control=control)
                    sym = value_for(summary_rows, task, model, final_budget, "condition_symbolic_exact", write_mode=write_mode, control=control)
                    hit = value_for(summary_rows, task, model, final_budget, "symbolic_binding_hit_1_rate", write_mode=write_mode, control=control)
                    available = value_for(summary_rows, task, model, final_budget, "exact_match_available_rate", write_mode=write_mode, control=control)
                    ambiguous = value_for(summary_rows, task, model, final_budget, "ambiguous_exact_match_rate", write_mode=write_mode, control=control)
                    slot = value_for(summary_rows, task, model, final_budget, "condition_symbolic_slot_accuracy", write_mode=write_mode, control=control)
                    value = value_for(summary_rows, task, model, final_budget, "condition_symbolic_value_accuracy", write_mode=write_mode, control=control)
                    ce_gap = value_for(summary_rows, task, model, final_budget, "ce_vs_symbolic_gap", write_mode=write_mode, control=control)
                    ret_gap = value_for(summary_rows, task, model, final_budget, "retrieval_vs_symbolic_gap", write_mode=write_mode, control=control)
                    lines.append(
                        "| {task} | {write} | {model} | {control} | {ce} | {sym} | {hit} | {available} | {ambiguous} | {slot} | {value} | {ce_gap} | {ret_gap} |".format(
                            task=task,
                            write=write_mode,
                            model=model,
                            control=control,
                            ce="" if ce is None else f"{ce:.4f}",
                            sym="" if sym is None else f"{sym:.4f}",
                            hit="" if hit is None else f"{hit:.4f}",
                            available="" if available is None else f"{available:.4f}",
                            ambiguous="" if ambiguous is None else f"{ambiguous:.4f}",
                            slot="" if slot is None else f"{slot:.4f}",
                            value="" if value is None else f"{value:.4f}",
                            ce_gap="" if ce_gap is None else f"{ce_gap:.4f}",
                            ret_gap="" if ret_gap is None else f"{ret_gap:.4f}",
                        )
                    )

    target_task = "conditional_contrastive" if "conditional_contrastive" in conditional_tasks else conditional_tasks[0] if conditional_tasks else ""
    epmem_ce = value_for(summary_rows, target_task, "epmem", final_budget, "answer_exact") if target_task else None
    epmem_symbolic = value_for(summary_rows, target_task, "epmem", final_budget, "condition_symbolic_exact") if target_task else None
    epmem_hit = value_for(summary_rows, target_task, "epmem", final_budget, "symbolic_binding_hit_1_rate") if target_task else None
    epmem_available = value_for(summary_rows, target_task, "epmem", final_budget, "exact_match_available_rate") if target_task else None
    hpm_ce = value_for(summary_rows, target_task, "hpm_lite", final_budget, "answer_exact") if target_task else None
    hpm_symbolic = value_for(summary_rows, target_task, "hpm_lite", final_budget, "condition_symbolic_exact") if target_task else None
    hebb_ce = value_for(summary_rows, target_task, "hebbian", final_budget, "answer_exact") if target_task else None
    hebb_symbolic = value_for(summary_rows, target_task, "hebbian", final_budget, "condition_symbolic_exact") if target_task else None
    local_ce = value_for(summary_rows, target_task, "local", final_budget, "answer_exact") if target_task else None
    local_symbolic = value_for(summary_rows, target_task, "local", final_budget, "condition_symbolic_exact") if target_task else None
    lines.extend(["", "### Symbolic Binding Verdict", ""])
    if target_task:
        lines.append("Did symbolic condition binding hit 1.0?")
        if epmem_symbolic is not None and epmem_symbolic >= 0.99:
            lines.append("Yes. Symbolic condition binding is essentially solved for epmem.")
        elif epmem_symbolic is not None:
            lines.append(f"No. epmem symbolic condition exact is {epmem_symbolic:.3f}.")
        if epmem_ce is not None and epmem_symbolic is not None and epmem_hit is not None and epmem_available is not None:
            lines.append(f"At budget {final_budget} on {target_task} with fact_token: epmem CE exact = {epmem_ce:.3f}; epmem symbolic condition exact = {epmem_symbolic:.3f}; epmem symbolic_binding_hit_1_rate = {epmem_hit:.3f}; exact_match_available_rate = {epmem_available:.3f}.")
        if hpm_ce is not None and hpm_symbolic is not None:
            lines.append(f"hpm_lite CE exact = {hpm_ce:.3f}; hpm_lite symbolic condition exact = {hpm_symbolic:.3f}.")
        if hebb_ce is not None and hebb_symbolic is not None:
            lines.append(f"hebbian CE exact = {hebb_ce:.3f}; hebbian symbolic condition exact = {hebb_symbolic:.3f}.")
        if local_ce is not None and local_symbolic is not None:
            lines.append(f"local sanity: CE exact = {local_ce:.3f}; local + external symbolic readout exact = {local_symbolic:.3f}.")
        if epmem_symbolic is not None and epmem_available is not None:
            if epmem_symbolic >= 0.99 and epmem_ce is not None and epmem_ce < 0.8:
                lines.append("Symbolic condition binding is essentially solved. Memory contains the correct information. CE decoding/read-use is the bottleneck.")
                lines.append("This is strong evidence that generic next-token CE decoding is the wrong readout for condition-binding memory.")
                lines.append("If symbolic binding hits 1.0 while CE remains low, this is a write-up-worthy result.")
            elif epmem_symbolic < 0.99 and epmem_available >= 0.99:
                lines.append("Memory contains matching slots, but symbolic slot selection/parsing is broken.")
            elif epmem_available < 0.99:
                lines.append("The memory writer/parser is failing to store the needed key-condition-value slot.")
    lines.append("")

    section = "\n".join(lines)
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## Condition Binding Structured Readout\n"
        index = old.find(marker)
        if index >= 0:
            old = old[:index].rstrip()
        path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
    else:
        path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tasks = parse_str_list(args.tasks, set(TASKS))
    models = parse_str_list(args.models, set(MODELS))
    write_modes = parse_write_modes(args.write_modes)
    controls = parse_str_list(args.controls, set(CONTROLS))
    budgets = sorted(set(parse_int_list(args.budgets)))
    seeds = parse_int_list(args.seeds)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    raw_rows: List[Dict[str, object]] = []
    leak_checks = 0
    for task in tasks:
        for seed in seeds:
            leak_checks += verify_no_answer_in_memory(
                task=task,
                seq_len=args.seq_len,
                window=args.window,
                seed=seed,
                batch_size=args.batch_size,
                num_facts=args.num_facts,
                write_modes=write_modes,
            )
        for write_mode in write_modes:
            for model_name in models:
                for seed in seeds:
                    train_and_evaluate(
                        task=task,
                        model_name=model_name,
                        write_mode=write_mode,
                        seed=seed,
                        budgets=budgets,
                        controls=controls,
                        args=args,
                        device=device,
                        raw_rows=raw_rows,
                        out_dir=out_dir,
                    )

    summary_rows = flush_outputs(out_dir, raw_rows)
    update_results_md(Path("results.md"), summary_rows, args, leak_checks)
    print("wrote runs/readuse_raw.csv, runs/readuse_summary.csv, and results.md")


if __name__ == "__main__":
    main()
