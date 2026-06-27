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

from hpm_lite.data import FactRecallConfig, FactRecallDataset, QUERY
from hpm_lite.evaluate import evaluate_batches
from hpm_lite.metrics import answer_cross_entropy
from hpm_lite.train import TinyAdamW, make_model
from hpm_lite.utils import ensure_dir, resolve_device, set_seed
from hpm_lite.write_modes import apply_write_mode, first_positions, parse_write_modes


MODELS = ["local", "epmem", "hpm_lite", "hebbian"]
TASKS = ["kv", "coexisting", "conditional", "longhop"]
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
    parser = argparse.ArgumentParser(description="Run MEMFAIL-Lite diagnostic controls.")
    parser.add_argument("--tasks", type=str, default="kv,coexisting,conditional,longhop")
    parser.add_argument("--models", type=str, default="local,epmem,hpm_lite,hebbian")
    parser.add_argument("--write-modes", type=str, default="oracle,fact_token,random_write")
    parser.add_argument("--controls", type=str, default="normal,no_retrieval,shuffled_values,random_keys")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seq-lens", type=str, default="512")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--lambda-ret", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=2)
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
            repeated_keys=False,
            similar_values=False,
            distractor_fact_spans=0,
            query_key_noise_only=False,
            fact_order="random",
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
    checked = 0
    for offset in range(2):
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
                leaked = torch.isin(valid_positions.reshape(-1), answer_input_positions[b])
                if torch.any(leaked):
                    raise AssertionError("future answer token leaked into memory writes")
        checked += batch_size
        dataset = dataset_for(task, seq_len, window, seed + offset + 1, num_facts)
    return checked


def train_once(
    task: str,
    model_name: str,
    write_mode: str,
    seq_len: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, float]:
    set_seed(seed)
    model_args = SimpleNamespace(
        model=model_name,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        window=args.window,
        seq_len=seq_len,
    )
    model = make_model(model_args, device)
    optimizer = TinyAdamW(model.parameters(), lr=args.lr)
    dataset = dataset_for(task, seq_len, args.window, seed, args.num_facts)
    start = time.perf_counter()

    for _ in range(args.steps):
        model.train()
        batch = dataset.sample_batch(args.batch_size, device=device)
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
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for {task}/{model_name}/{write_mode}/seed{seed}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    elapsed = time.perf_counter() - start
    train_eps = (args.steps * args.batch_size) / max(elapsed, 1.0e-9)
    return model, train_eps


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def summarize(raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str, int, str], List[Dict[str, object]]] = {}
    for row in raw_rows:
        key = (
            str(row["task"]),
            str(row["write_mode"]),
            str(row["model"]),
            int(row["seq_len"]),
            str(row["memory_control"]),
        )
        grouped.setdefault(key, []).append(row)

    metrics = [
        "answer_exact",
        "answer_ce",
        "retrieval_top1",
        "retrieval_topk",
        "reasoning_success_given_retrieval",
        "examples_per_sec",
        "train_examples_per_sec",
        "avg_written_slots",
        "true_fact_written_rate",
        "false_write_rate",
        "missed_fact_rate",
    ]
    summary_rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        task, write_mode, model, seq_len, control = key
        rows = grouped[key]
        out: Dict[str, object] = {
            "task": task,
            "write_mode": write_mode,
            "model": model,
            "seq_len": seq_len,
            "memory_control": control,
            "n": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric, "") != ""]
            if values:
                out[f"{metric}_mean"] = statistics.mean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
        summary_rows.append(out)
    return summary_rows


def avg_exact(summary_rows: List[Dict[str, object]], task: str, model: str, write_mode: str = "fact_token") -> float | None:
    values = [
        float(row["answer_exact_mean"])
        for row in summary_rows
        if row["task"] == task
        and row["model"] == model
        and row["write_mode"] == write_mode
        and row["memory_control"] == "normal"
    ]
    return statistics.mean(values) if values else None


def model_status(summary_rows: List[Dict[str, object]], task: str) -> str:
    parts = []
    for model in MODELS:
        value = avg_exact(summary_rows, task, model)
        if value is None:
            continue
        if value >= 0.8:
            label = "passes"
        elif value >= 0.5:
            label = "is borderline"
        else:
            label = "fails"
        parts.append(f"{model} {label} ({value:.3f})")
    return "; ".join(parts) if parts else "no fact-token normal rows"


def verdict_lines(summary_rows: List[Dict[str, object]]) -> List[str]:
    lines = [
        f"- Exact recall: {model_status(summary_rows, 'kv')}.",
        f"- Coexisting facts: {model_status(summary_rows, 'coexisting')}.",
        f"- Conditional facts: {model_status(summary_rows, 'conditional')}.",
        f"- Long-hop composition: {model_status(summary_rows, 'longhop')}.",
    ]
    hpm_deltas = []
    for task in TASKS:
        hpm = avg_exact(summary_rows, task, "hpm_lite")
        epmem = avg_exact(summary_rows, task, "epmem")
        if hpm is not None and epmem is not None:
            hpm_deltas.append((task, hpm - epmem))
    if hpm_deltas:
        strongest = max(hpm_deltas, key=lambda item: abs(item[1]))
        lines.append(
            f"- HPM-Lite vs epmem: largest fact-token normal delta was {strongest[1] * 100:.1f} points on `{strongest[0]}`."
        )
    hebb_kv = avg_exact(summary_rows, "kv", "hebbian")
    hebb_other = [
        value
        for task in ["coexisting", "conditional", "longhop"]
        for value in [avg_exact(summary_rows, task, "hebbian")]
        if value is not None
    ]
    if hebb_kv is not None and hebb_other:
        lines.append(
            f"- Hebbian: clean KV exact was {hebb_kv:.3f}; the mean across interference/qualifier/composition tasks was {statistics.mean(hebb_other):.3f}. "
            "Treat clean long-hop success cautiously; the prior hard audit already showed this baseline is not robust under random-order/corruption controls."
        )
    lines.append("- This MEMFAIL-Lite run keeps distractor complexity off; failures here are diagnostic, not a full robustness audit.")
    return lines


def format_float(value: object) -> str:
    if value == "":
        return ""
    return f"{float(value):.4f}"


def update_results_md(path: Path, summary_rows: List[Dict[str, object]], args: argparse.Namespace, leak_checks: int) -> None:
    lines = [
        "## MEMFAIL-Lite Diagnostics",
        "",
        "This section refactors the plan from a single clean recall task toward small diagnostic tasks that separate memory failure modes. "
        "It keeps the existing Stage 2 KV results above intact and does not add a learned writer, JEPA, ANN, Priming, GKA, or full HPM routing.",
        "",
        f"Tasks: `{args.tasks}`. Models: `{args.models}`. Write modes: `{args.write_modes}`. Controls: `{args.controls}`.",
        f"Seeds: `{args.seeds}`. Sequence lengths: `{args.seq_lens}`. Window: `{args.window}`.",
        f"Steps: `{args.steps}`. Batch size: `{args.batch_size}`. Eval batches: `{args.eval_batches}`. Top-k: `{args.top_k}`.",
        f"Device request: `{args.device}`. Platform: `{platform.platform()}`. Torch: `{torch.__version__}`.",
        f"Leak checks passed on `{leak_checks}` generated examples across all requested write modes; memory writes stayed pre-query and excluded future answer tokens.",
        "",
        "Command:",
        "",
        f"- `python scripts/run_memfail_lite.py --steps {args.steps} --batch-size {args.batch_size} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device} --eval-batches {args.eval_batches} --seeds {args.seeds} --seq-lens {args.seq_lens}`",
        "",
        "Raw and summarized outputs: `runs/memfail_raw.csv`, `runs/memfail_summary.csv`.",
        "",
        "### Compact Summary",
        "",
        "| task | write | model | seq_len | control | exact | CE | ret top1 | ret topk | use if ret | slots | true write | false write | missed | ex/sec |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {task} | {write} | {model} | {seq_len} | {control} | {exact} | {ce} | {ret1} | {retk} | {reason} | {slots} | {true} | {false} | {missed} | {eps} |".format(
                task=row["task"],
                write=row["write_mode"],
                model=row["model"],
                seq_len=row["seq_len"],
                control=row["memory_control"],
                exact=format_float(row["answer_exact_mean"]),
                ce=format_float(row["answer_ce_mean"]),
                ret1=format_float(row["retrieval_top1_mean"]),
                retk=format_float(row["retrieval_topk_mean"]),
                reason=format_float(row["reasoning_success_given_retrieval_mean"]),
                slots=format_float(row["avg_written_slots_mean"]),
                true=format_float(row["true_fact_written_rate_mean"]),
                false=format_float(row["false_write_rate_mean"]),
                missed=format_float(row["missed_fact_rate_mean"]),
                eps=format_float(row["examples_per_sec_mean"]),
            )
        )

    lines.extend(["", "### MEMFAIL-Lite Verdict", ""])
    lines.extend(verdict_lines(summary_rows))
    lines.extend(
        [
            "",
            "Interpretation guardrail: the table above is a small CPU diagnostic run. It is meant to show whether the tasks and controls execute cleanly and expose separable failure modes, not to claim final model quality.",
            "",
        ]
    )
    section = "\n".join(lines)
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## MEMFAIL-Lite Diagnostics\n"
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
    seeds = parse_int_list(args.seeds)
    seq_lens = parse_int_list(args.seq_lens)
    device = resolve_device(args.device)
    ensure_dir(args.out_dir)

    raw_rows: List[Dict[str, object]] = []
    leak_checks = 0
    for task in tasks:
        for seq_len in seq_lens:
            for seed in seeds:
                leak_checks += verify_no_answer_in_memory(
                    task=task,
                    seq_len=seq_len,
                    window=args.window,
                    seed=seed,
                    batch_size=args.batch_size,
                    num_facts=args.num_facts,
                    write_modes=write_modes,
                )
                for write_mode in write_modes:
                    for model_name in models:
                        model, train_eps = train_once(task, model_name, write_mode, seq_len, seed, args, device)
                        for control in controls:
                            eval_dataset = dataset_for(
                                task=task,
                                seq_len=seq_len,
                                window=args.window,
                                seed=200_000 + seed + seq_len,
                                num_facts=args.num_facts,
                            )
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
                            raw_rows.append(
                                {
                                    "task": task,
                                    "write_mode": write_mode,
                                    "model": model_name,
                                    "seq_len": seq_len,
                                    "seed": seed,
                                    "memory_control": control,
                                    "answer_exact": metrics["answer_exact"],
                                    "answer_ce": metrics["answer_ce"],
                                    "retrieval_top1": metrics.get("retrieval_top1", ""),
                                    "retrieval_topk": metrics.get("retrieval_topk", ""),
                                    "reasoning_success_given_retrieval": metrics.get(
                                        "reasoning_success_given_retrieval", ""
                                    ),
                                    "examples_per_sec": metrics["examples_per_sec"],
                                    "train_examples_per_sec": train_eps,
                                    "avg_written_slots": metrics["avg_written_slots"],
                                    "true_fact_written_rate": metrics["true_fact_written_rate"],
                                    "false_write_rate": metrics["false_write_rate"],
                                    "missed_fact_rate": metrics["missed_fact_rate"],
                                }
                            )

    summary_rows = summarize(raw_rows)
    raw_columns = [
        "task",
        "write_mode",
        "model",
        "seq_len",
        "seed",
        "memory_control",
        "answer_exact",
        "answer_ce",
        "retrieval_top1",
        "retrieval_topk",
        "reasoning_success_given_retrieval",
        "examples_per_sec",
        "train_examples_per_sec",
        "avg_written_slots",
        "true_fact_written_rate",
        "false_write_rate",
        "missed_fact_rate",
    ]
    summary_columns = [
        "task",
        "write_mode",
        "model",
        "seq_len",
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
        "examples_per_sec_mean",
        "examples_per_sec_std",
        "train_examples_per_sec_mean",
        "train_examples_per_sec_std",
        "avg_written_slots_mean",
        "avg_written_slots_std",
        "true_fact_written_rate_mean",
        "true_fact_written_rate_std",
        "false_write_rate_mean",
        "false_write_rate_std",
        "missed_fact_rate_mean",
        "missed_fact_rate_std",
    ]
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "memfail_raw.csv", raw_rows, raw_columns)
    write_csv(out_dir / "memfail_summary.csv", summary_rows, summary_columns)
    update_results_md(Path("results.md"), summary_rows, args, leak_checks)
    print("wrote runs/memfail_raw.csv, runs/memfail_summary.csv, and results.md")


if __name__ == "__main__":
    main()
