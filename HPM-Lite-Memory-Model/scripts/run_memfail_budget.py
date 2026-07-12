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


TASKS = ["kv", "coexisting", "conditional", "longhop"]
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
    parser = argparse.ArgumentParser(description="Run MEMFAIL-Lite training-budget sweep.")
    parser.add_argument("--tasks", type=str, default="kv,coexisting,conditional,longhop")
    parser.add_argument("--models", type=str, default="local,epmem,hpm_lite,hebbian")
    parser.add_argument("--write-modes", type=str, default="oracle,fact_token")
    parser.add_argument("--controls", type=str, default="normal,no_retrieval,shuffled_values,random_keys")
    parser.add_argument("--budgets", type=str, default="3,10,30,100,300")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=5)
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
            int(row["budget_steps"]),
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
        "train_wall_time_sec",
        "avg_written_slots",
        "true_fact_written_rate",
        "false_write_rate",
        "missed_fact_rate",
    ]
    summary_rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        task, write_mode, model, budget, control = key
        rows = grouped[key]
        out: Dict[str, object] = {
            "task": task,
            "write_mode": write_mode,
            "model": model,
            "budget_steps": budget,
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


RAW_COLUMNS = [
    "task",
    "write_mode",
    "model",
    "seed",
    "budget_steps",
    "seq_len",
    "memory_control",
    "answer_exact",
    "answer_ce",
    "retrieval_top1",
    "retrieval_topk",
    "reasoning_success_given_retrieval",
    "examples_per_sec",
    "train_examples_per_sec",
    "train_wall_time_sec",
    "avg_written_slots",
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
    "examples_per_sec_mean",
    "examples_per_sec_std",
    "train_examples_per_sec_mean",
    "train_examples_per_sec_std",
    "train_wall_time_sec_mean",
    "train_wall_time_sec_std",
    "avg_written_slots_mean",
    "avg_written_slots_std",
    "true_fact_written_rate_mean",
    "true_fact_written_rate_std",
    "false_write_rate_mean",
    "false_write_rate_std",
    "missed_fact_rate_mean",
    "missed_fact_rate_std",
]


def flush_outputs(out_dir: Path, raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows = summarize(raw_rows)
    write_csv(out_dir / "memfail_budget_raw.csv", raw_rows, RAW_COLUMNS)
    write_csv(out_dir / "memfail_budget_summary.csv", summary_rows, SUMMARY_COLUMNS)
    return summary_rows


def train_and_evaluate_config(
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
    optimizer = TinyAdamW(model.parameters(), lr=args.lr)
    train_dataset = dataset_for(task, args.seq_len, args.window, seed, args.num_facts)
    budget_set = set(budgets)
    max_budget = max(budgets)
    start = time.perf_counter()

    for step in range(1, max_budget + 1):
        model.train()
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
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for {task}/{model_name}/{write_mode}/seed{seed}/step{step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step not in budget_set:
            continue

        train_wall = time.perf_counter() - start
        train_eps = (step * args.batch_size) / max(train_wall, 1.0e-9)
        for control in controls:
            eval_dataset = dataset_for(
                task=task,
                seq_len=args.seq_len,
                window=args.window,
                seed=200_000 + seed + args.seq_len + step,
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
                    "seed": seed,
                    "budget_steps": step,
                    "seq_len": args.seq_len,
                    "memory_control": control,
                    "answer_exact": metrics["answer_exact"],
                    "answer_ce": metrics["answer_ce"],
                    "retrieval_top1": metrics.get("retrieval_top1", ""),
                    "retrieval_topk": metrics.get("retrieval_topk", ""),
                    "reasoning_success_given_retrieval": metrics.get("reasoning_success_given_retrieval", ""),
                    "examples_per_sec": metrics["examples_per_sec"],
                    "train_examples_per_sec": train_eps,
                    "train_wall_time_sec": train_wall,
                    "avg_written_slots": metrics["avg_written_slots"],
                    "true_fact_written_rate": metrics["true_fact_written_rate"],
                    "false_write_rate": metrics["false_write_rate"],
                    "missed_fact_rate": metrics["missed_fact_rate"],
                }
            )
        flush_outputs(out_dir, raw_rows)
        print(
            f"done task={task} model={model_name} write={write_mode} seed={seed} budget={step}",
            flush=True,
        )


def format_float(value: object) -> str:
    if value == "":
        return ""
    return f"{float(value):.4f}"


def exact_for(
    rows: List[Dict[str, object]],
    task: str,
    model: str,
    budget: int,
    write_mode: str = "fact_token",
    control: str = "normal",
) -> float | None:
    values = [
        float(row["answer_exact_mean"])
        for row in rows
        if row["task"] == task
        and row["model"] == model
        and int(row["budget_steps"]) == budget
        and row["write_mode"] == write_mode
        and row["memory_control"] == control
        and row["answer_exact_mean"] != ""
    ]
    return statistics.mean(values) if values else None


def metric_for(
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


def verdict_lines(summary_rows: List[Dict[str, object]], budgets: List[int]) -> List[str]:
    final_budget = max(budgets)
    lines: List[str] = []
    co_ret = metric_for(summary_rows, "coexisting", "epmem", final_budget, "retrieval_topk")
    co_exact = exact_for(summary_rows, "coexisting", "epmem", final_budget)
    if co_ret is not None and co_exact is not None:
        label = "read/use composition failure" if co_ret >= 0.8 and co_exact < 0.5 else "training-sensitive or unresolved"
        lines.append(f"- Coexisting at {final_budget} steps: epmem top-k {co_ret:.3f}, exact {co_exact:.3f}; mark as {label}.")

    cond_ret = metric_for(summary_rows, "conditional", "epmem", final_budget, "retrieval_topk")
    cond_exact = exact_for(summary_rows, "conditional", "epmem", final_budget)
    if cond_ret is not None and cond_exact is not None:
        label = "qualifier-use failure" if cond_ret >= 0.8 and cond_exact < 0.5 else "training-sensitive or unresolved"
        lines.append(f"- Conditional at {final_budget} steps: epmem top-k {cond_ret:.3f}, exact {cond_exact:.3f}; mark as {label}.")

    deltas = []
    for budget in budgets:
        for task in TASKS:
            hpm = exact_for(summary_rows, task, "hpm_lite", budget)
            epmem = exact_for(summary_rows, task, "epmem", budget)
            if hpm is not None and epmem is not None:
                deltas.append((budget, task, hpm - epmem))
    if deltas:
        consistent = [
            (budget, task, delta)
            for budget, task, delta in deltas
            if delta >= 0.05
        ]
        strongest = max(deltas, key=lambda item: item[2])
        if len(consistent) < max(2, len(deltas) // 4):
            lines.append(
                f"- HPM-Lite is not justified yet: strongest observed delta was {strongest[2] * 100:.1f} points on `{strongest[1]}` at {strongest[0]} steps, not a consistent 5-10 point win."
            )
        else:
            lines.append("- HPM-Lite shows some repeated separation from epmem, but this still needs a larger confirmatory run.")

    hebb_values = {
        task: exact_for(summary_rows, task, "hebbian", final_budget)
        for task in TASKS
    }
    if all(value is not None for value in hebb_values.values()):
        lines.append(
            "- Hebbian at final budget: "
            + ", ".join(f"{task} {value:.3f}" for task, value in hebb_values.items() if value is not None)
            + "."
        )

    no_ret = [
        float(row["answer_exact_mean"])
        for row in summary_rows
        if row["memory_control"] == "no_retrieval"
        and int(row["budget_steps"]) == final_budget
        and row["write_mode"] == "fact_token"
        and row["answer_exact_mean"] != ""
    ]
    if no_ret:
        lines.append(f"- No-retrieval control at final budget averaged {statistics.mean(no_ret):.3f} exact.")

    hurt_rows = []
    for task in TASKS:
        for model in ["epmem", "hpm_lite", "hebbian"]:
            normal = exact_for(summary_rows, task, model, final_budget)
            shuffled = exact_for(summary_rows, task, model, final_budget, control="shuffled_values")
            random_keys = exact_for(summary_rows, task, model, final_budget, control="random_keys")
            if normal is not None and shuffled is not None and random_keys is not None:
                hurt_rows.append((normal - shuffled, normal - random_keys))
    if hurt_rows:
        shuffled_hurt = statistics.mean(item[0] for item in hurt_rows)
        random_hurt = statistics.mean(item[1] for item in hurt_rows)
        lines.append(f"- Controls still hurt: shuffled-values mean drop {shuffled_hurt * 100:.1f} points, random-keys mean drop {random_hurt * 100:.1f} points at final budget.")

    lines.append("- If any failure vanishes at larger budgets, treat it as insufficient training rather than an architecture failure.")
    return lines


def update_results_md(path: Path, summary_rows: List[Dict[str, object]], args: argparse.Namespace, leak_checks: int) -> None:
    budgets = parse_int_list(args.budgets)
    lines = [
        "## Training Budget Sweep",
        "",
        "This sweep checks whether MEMFAIL-Lite failures disappear with more optimization. It does not add learned writing, distractors, JEPA, ANN, GKA, Priming, RL, graph memory, or a new architecture.",
        "",
        f"Tasks: `{args.tasks}`. Models: `{args.models}`. Write modes: `{args.write_modes}`. Controls: `{args.controls}`.",
        f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Seq len/window: `{args.seq_len}` / `{args.window}`.",
        f"Model size: d_model `{args.d_model}`, layers `{args.layers}`, heads `{args.heads}`, batch size `{args.batch_size}`.",
        f"Eval batches: `{args.eval_batches}`. Device request: `{args.device}`. Platform: `{platform.platform()}`. Torch: `{torch.__version__}`.",
        f"Leak checks passed on `{leak_checks}` generated examples; memory writes stayed pre-query and excluded future answer tokens.",
        "",
        "Command:",
        "",
        f"- `python scripts/run_memfail_budget.py --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device}`",
        "",
        "Raw and summarized outputs: `runs/memfail_budget_raw.csv`, `runs/memfail_budget_summary.csv`.",
        "",
        "### Fact-Token Normal Accuracy By Budget",
        "",
        "| task | model | 3 | 10 | 30 | 100 | 300 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        for model in MODELS:
            values = []
            for budget in budgets:
                value = exact_for(summary_rows, task, model, budget)
                values.append("" if value is None else f"{value:.4f}")
            lines.append(f"| {task} | {model} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "### Final-Budget Controls",
            "",
            "| task | model | normal | no_retrieval | shuffled_values | random_keys |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    final_budget = max(budgets)
    for task in TASKS:
        for model in MODELS:
            values = []
            for control in CONTROLS:
                value = exact_for(summary_rows, task, model, final_budget, control=control)
                values.append("" if value is None else f"{value:.4f}")
            lines.append(f"| {task} | {model} | " + " | ".join(values) + " |")

    lines.extend(["", "### Budget Verdict", ""])
    lines.extend(verdict_lines(summary_rows, budgets))
    lines.append("")

    section = "\n".join(lines)
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## Training Budget Sweep\n"
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
                    train_and_evaluate_config(
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
    print("wrote runs/memfail_budget_raw.csv, runs/memfail_budget_summary.csv, and results.md")


if __name__ == "__main__":
    main()
