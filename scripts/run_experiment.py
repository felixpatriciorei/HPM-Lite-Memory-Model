from __future__ import annotations

import argparse
import csv
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpm_lite.train import run_training


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small HPM-Lite model comparison.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs/experiment")
    return parser


def command_line_for(model: str, args: argparse.Namespace, seed: int) -> str:
    return (
        "python -m hpm_lite.train "
        f"--model {model} --task kv --steps {args.steps} --seq-len {args.seq_len} "
        f"--window {args.window} --batch-size {args.batch_size} --d-model {args.d_model} "
        f"--layers {args.layers} --heads {args.heads} --seed {seed} --device {args.device}"
    )


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "eval_answer_exact",
        "eval_answer_ce",
        "eval_retrieval_top1",
        "eval_retrieval_topk",
        "examples_per_sec_recent",
        "parameters",
        "memory_slots_per_sample",
        "train_wall_time_sec",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_results_md(path: Path, rows: List[Dict[str, object]], commands: List[str], args: argparse.Namespace) -> None:
    local = next((row for row in rows if row["model"] == "local"), None)
    memory_rows = [row for row in rows if row["model"] in {"epmem", "hpm_lite", "hebbian"}]
    best_memory = max(memory_rows, key=lambda row: float(row.get("eval_answer_exact", 0.0))) if memory_rows else None

    interpretation = "No comparison was available."
    if local and best_memory:
        gain = float(best_memory.get("eval_answer_exact", 0.0)) - float(local.get("eval_answer_exact", 0.0))
        time_ratio = float(best_memory.get("train_wall_time_sec", 1.0)) / max(float(local.get("train_wall_time_sec", 1.0)), 1.0e-9)
        if float(local.get("eval_answer_exact", 0.0)) > 0.8:
            interpretation = "Local attention already solved this setting; increase gap, value count, distractors, or use two-hop."
        elif gain < 0.05 and time_ratio > 1.5:
            interpretation = "Weak result: the memory model did not beat local by 5 percentage points at reasonable cost."
        elif gain >= 0.05:
            interpretation = (
                f"Interesting sanity-check result: best memory model beat local by {gain * 100:.1f} percentage points "
                f"with a {time_ratio:.2f}x train-time ratio."
            )
        else:
            interpretation = "Memory did not clearly improve exact recall in this run."

    lines = [
        "# HPM-Lite Results",
        "",
        "## What Was Run",
        "",
        f"Task: `kv`, seq_len: `{args.seq_len}`, window: `{args.window}`, steps: `{args.steps}`, batch_size: `{args.batch_size}`.",
        f"Hardware/device request: `{args.device}`. Platform: `{platform.platform()}`. Torch: `{torch.__version__}`.",
        "This is a small comparison run; use more steps before treating the result as evidence.",
        "",
        "## Commands",
        "",
    ]
    lines.extend(f"- `{command}`" for command in commands)
    lines.extend(["", "## Metrics", "", "| model | exact | answer CE | ret top1 | ret topk | params | wall time |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(
            "| {model} | {exact:.4f} | {ce:.4f} | {ret1} | {retk} | {params} | {wall:.2f}s |".format(
                model=row["model"],
                exact=float(row.get("eval_answer_exact", 0.0)),
                ce=float(row.get("eval_answer_ce", 0.0)),
                ret1="" if row.get("eval_retrieval_top1", "") == "" else f"{float(row.get('eval_retrieval_top1')):.4f}",
                retk="" if row.get("eval_retrieval_topk", "") == "" else f"{float(row.get('eval_retrieval_topk')):.4f}",
                params=int(row.get("parameters", 0)),
                wall=float(row.get("train_wall_time_sec", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            "Pass/fail rule: HPM-lite/epmem is interesting only if it beats local by a clear margin on long-gap recall. "
            "A gain under 5 percentage points with much worse compute is weak.",
            "",
            "## Known Limitations",
            "",
            "- Oracle memory writes are enabled by default; this does not test learned write policies.",
            "- The memory path uses token-level fact representations, so this is a mechanism sanity check.",
            "- This is not a practical Transformer successor yet.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    rows: List[Dict[str, object]] = []
    commands: List[str] = []

    for offset, model in enumerate(["local", "epmem", "hpm_lite", "hebbian"]):
        seed = args.seed + offset
        commands.append(command_line_for(model, args, seed))
        train_args = SimpleNamespace(
            model=model,
            task="kv",
            seq_len=args.seq_len,
            window=args.window,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_every=max(1, args.steps),
            eval_batches=5,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            lr=3.0e-4,
            seed=seed,
            device=args.device,
            lambda_ret=0.1,
            top_k=1,
            memory_control="normal",
            oracle_memory=True,
            out_dir=args.out_dir,
            save_checkpoint=True,
        )
        rows.append(run_training(train_args))

    write_summary_csv(Path("runs") / "summary.csv", rows)
    write_results_md(Path("results.md"), rows, commands, args)
    print("wrote runs/summary.csv and results.md")


if __name__ == "__main__":
    main()
