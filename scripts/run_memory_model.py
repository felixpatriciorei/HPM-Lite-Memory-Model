from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from hpm_lite.train import run_training


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact HPM-Lite Memory Model comparison from the project brief."
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-facts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs/memory_model")
    parser.add_argument("--memory-null-slot", action="store_true")
    parser.add_argument("--null-score-init", type=float, default=0.0)
    parser.add_argument("--write-mode", choices=["oracle", "learned"], default="oracle")
    parser.add_argument("--lambda-writer", type=float, default=0.1)
    parser.add_argument("--learned-writer-teacher-forcing-steps", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = []
    for offset, model in enumerate(["local", "hpm_lite"]):
        rows.append(
            run_training(
                SimpleNamespace(
                    model=model,
                    task="kv",
                    seq_len=args.seq_len,
                    window=args.window,
                    batch_size=args.batch_size,
                    steps=args.steps,
                    eval_every=max(1, args.steps),
                    eval_batches=10,
                    d_model=args.d_model,
                    layers=args.layers,
                    heads=args.heads,
                    lr=3.0e-4,
                    seed=args.seed + offset,
                    device=args.device,
                    lambda_ret=0.1,
                    lambda_writer=args.lambda_writer,
                    learned_writer_teacher_forcing_steps=args.learned_writer_teacher_forcing_steps,
                    top_k=1,
                    memory_null_slot=args.memory_null_slot,
                    null_score_init=args.null_score_init,
                    memory_control="normal",
                    write_mode=args.write_mode if model == "hpm_lite" else "oracle",
                    oracle_memory=True,
                    num_facts=args.num_facts,
                    repeated_keys=False,
                    similar_values=False,
                    distractor_fact_spans=0,
                    query_key_noise_only=False,
                    fact_order="random",
                    out_dir=args.out_dir,
                    save_checkpoint=True,
                )
            )
        )

    out_path = Path(args.out_dir) / "memory_model_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "eval_answer_exact",
        "eval_answer_ce",
        "eval_retrieval_top1",
        "eval_retrieval_topk",
        "eval_avg_written_slots",
        "eval_true_fact_written_rate",
        "eval_false_write_rate",
        "eval_missed_fact_rate",
        "parameters",
        "train_wall_time_sec",
        "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    local = next(row for row in rows if row["model"] == "local")
    hpm = next(row for row in rows if row["model"] == "hpm_lite")
    gain = float(hpm.get("eval_answer_exact", 0.0)) - float(local.get("eval_answer_exact", 0.0))
    print(f"wrote {out_path}")
    print(f"local exact={float(local.get('eval_answer_exact', 0.0)):.4f}")
    print(f"hpm_lite exact={float(hpm.get('eval_answer_exact', 0.0)):.4f}")
    print(f"exact gain={gain:.4f}")


if __name__ == "__main__":
    main()
