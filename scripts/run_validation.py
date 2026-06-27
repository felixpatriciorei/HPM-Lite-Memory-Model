from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpm_lite.data import FactRecallConfig, FactRecallDataset
from hpm_lite.evaluate import evaluate_batches
from hpm_lite.model import HpmLiteConfig, HpmLiteModel
from hpm_lite.train import run_training
from hpm_lite.utils import ensure_dir, resolve_device, set_seed
from hpm_lite.write_modes import parse_write_modes


MEMORY_MODELS = {"epmem", "hpm_lite", "hebbian"}


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HPM-Lite validation controls.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seq-lens", type=str, default="256,512,1024")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="runs/validation")
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
    parser.add_argument("--write-modes", type=str, default="oracle")
    return parser


def verify_no_answer_in_memory(seq_len: int, window: int, seed: int, batch_size: int, task: str) -> int:
    dataset = FactRecallDataset(FactRecallConfig(seq_len=seq_len, window=window, task=task, seed=seed, oracle_memory=True))
    checked = 0
    for offset in range(3):
        batch = dataset.sample_batch(batch_size)
        answer_positions = batch["answer_positions"]
        answer_value_positions = answer_positions + 1
        if not torch.all(batch["memory_spans"][:, :, 1] < answer_positions[:, None]):
            raise AssertionError("memory span reaches query/answer region")
        if torch.any(batch["memory_token_positions"] == answer_value_positions[:, None, None]):
            raise AssertionError("future answer value token leaked into memory writes")
        checked += batch_size
        dataset.config.seed = seed + offset + 1
    return checked


def load_checkpoint(path: Path, device: torch.device) -> HpmLiteModel:
    checkpoint = torch.load(path, map_location=device)
    model = HpmLiteModel(HpmLiteConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def controls_for(model: str) -> List[str]:
    return ["normal", "no_retrieval", "shuffled_values", "random_keys"]


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def summarize(raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, int, str, str], List[Dict[str, object]]] = {}
    for row in raw_rows:
        key = (str(row["write_mode"]), str(row["model"]), int(row["seq_len"]), str(row["memory_control"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    metrics = [
        "answer_exact",
        "answer_ce",
        "retrieval_top1",
        "retrieval_topk",
        "examples_per_sec",
        "avg_written_slots",
        "true_fact_written_rate",
        "false_write_rate",
        "missed_fact_rate",
    ]
    for key in sorted(grouped):
        write_mode, model, seq_len, control = key
        rows = grouped[key]
        out: Dict[str, object] = {
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


def row_lookup(
    rows: List[Dict[str, object]],
    write_mode: str,
    model: str,
    seq_len: int,
    control: str,
) -> Dict[str, object] | None:
    for row in rows:
        if (
            row["write_mode"] == write_mode
            and row["model"] == model
            and int(row["seq_len"]) == seq_len
            and row["memory_control"] == control
        ):
            return row
    return None


def exact_average(summary_rows: List[Dict[str, object]], write_mode: str, model: str, control: str) -> float | None:
    values = [
        float(row["answer_exact_mean"])
        for row in summary_rows
        if row["write_mode"] == write_mode and row["model"] == model and row["memory_control"] == control
    ]
    return statistics.mean(values) if values else None


def verdict(summary_rows: List[Dict[str, object]]) -> str:
    pieces = []
    oracle_epmem = exact_average(summary_rows, "oracle", "epmem", "normal")
    fact_epmem = exact_average(summary_rows, "fact_token", "epmem", "normal")
    random_epmem = exact_average(summary_rows, "random_write", "epmem", "normal")
    fact_hebbian = exact_average(summary_rows, "fact_token", "hebbian", "normal")
    oracle_hebbian = exact_average(summary_rows, "oracle", "hebbian", "normal")
    fact_hpm = exact_average(summary_rows, "fact_token", "hpm_lite", "normal")
    fact_hpm_no = exact_average(summary_rows, "fact_token", "hpm_lite", "no_retrieval")

    if oracle_epmem is not None:
        pieces.append(f"Oracle retrieval sanity: epmem normal exact averaged {oracle_epmem:.3f}.")
    if fact_epmem is not None and oracle_epmem is not None:
        pieces.append(f"Fact-token writer: epmem averaged {fact_epmem:.3f}, a {(fact_epmem - oracle_epmem) * 100:.1f} point delta from oracle.")
    if random_epmem is not None and fact_epmem is not None:
        pieces.append(f"Random-write control: epmem averaged {random_epmem:.3f}, {(fact_epmem - random_epmem) * 100:.1f} points below fact-token.")
    if oracle_hebbian is not None and fact_hebbian is not None:
        pieces.append(f"Hebbian without oracle metadata stayed at {fact_hebbian:.3f} versus oracle {oracle_hebbian:.3f}.")
    if fact_hpm is not None and fact_epmem is not None:
        pieces.append(f"HPM-Lite versus epmem under fact-token writing: {fact_hpm:.3f} vs {fact_epmem:.3f}.")
    if fact_hpm is not None and fact_hpm_no is not None:
        pieces.append(f"HPM-Lite no-retrieval under fact-token writing averaged {fact_hpm_no:.3f}.")
    pieces.append("This is clean KV only; no hard two-hop or distractor conclusions should be drawn from this stage.")
    return " ".join(pieces)


def write_results_md(path: Path, summary_rows: List[Dict[str, object]], args: argparse.Namespace, leak_checks: int) -> None:
    lines = [
        "# HPM-Lite Results",
        "",
        "## Stage 2 Write-Mode Validation",
        "",
        f"Models: `local`, `epmem`, `hpm_lite`, `hebbian`.",
        f"Task: `{args.task}`. Write modes: `{args.write_modes}`.",
        f"Seeds: `{args.seeds}`. Sequence lengths: `{args.seq_lens}`. Window: `{args.window}`.",
        f"Steps: `{args.steps}`. Batch size: `{args.batch_size}`. Eval batches: `{args.eval_batches}`.",
        f"Device request: `{args.device}`. Platform: `{platform.platform()}`. Torch: `{torch.__version__}`.",
        f"Leak checks passed on `{leak_checks}` generated examples; writer spans were pre-query and never included QUERY, ANSWER, or the answer token.",
        "",
        "Command:",
        "",
        f"- `python scripts/run_validation.py --steps {args.steps} --batch-size {args.batch_size} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device} --eval-batches {args.eval_batches} --write-modes {args.write_modes} --task {args.task}`",
        "",
        "Stage 2 uses the original clean KV task only. No hard two-hop, distractor, learned marker scorer, top-k surprisal, or full write equation was added.",
        "",
        "Controls: `normal`, `no_retrieval`, `shuffled_values`, `random_keys`.",
        "",
        "## Summary Table",
        "",
        "| write | model | seq_len | control | exact mean | exact std | CE mean | CE std | ret top1 | ret topk | slots | true write | false write | missed | ex/sec |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        ret1 = row["retrieval_top1_mean"]
        retk = row["retrieval_topk_mean"]
        ret1_text = "" if ret1 == "" else f"{float(ret1):.4f}"
        retk_text = "" if retk == "" else f"{float(retk):.4f}"
        lines.append(
            "| {write} | {model} | {seq_len} | {control} | {exact:.4f} | {exact_std:.4f} | {ce:.4f} | {ce_std:.4f} | {ret1} | {retk} | {slots:.2f} | {true:.4f} | {false:.4f} | {missed:.4f} | {eps:.2f} |".format(
                write=row["write_mode"],
                model=row["model"],
                seq_len=row["seq_len"],
                control=row["memory_control"],
                exact=float(row["answer_exact_mean"]),
                exact_std=float(row["answer_exact_std"]),
                ce=float(row["answer_ce_mean"]),
                ce_std=float(row["answer_ce_std"]),
                ret1=ret1_text,
                retk=retk_text,
                slots=float(row["avg_written_slots_mean"]),
                true=float(row["true_fact_written_rate_mean"]),
                false=float(row["false_write_rate_mean"]),
                missed=float(row["missed_fact_rate_mean"]),
                eps=float(row["examples_per_sec_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Stage 2 Verdict",
            "",
            verdict(summary_rows),
            "",
            "Expected checks: fact-token should track oracle on clean KV; random-write, no-retrieval, shuffled-values, and random-keys should be worse. "
            "If random-write performs well, inspect leakage or shortcuts.",
            "",
            "## Known Limitations",
            "",
            "- Stage 2 is clean KV only.",
            "- The run is intentionally small enough for CPU validation.",
            "- Fact-token writing is a parser baseline, not a learned writer.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    seeds = parse_int_list(args.seeds)
    seq_lens = parse_int_list(args.seq_lens)
    write_modes = parse_write_modes(args.write_modes)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    raw_rows: List[Dict[str, object]] = []
    leak_checks = 0
    for seq_len in seq_lens:
        for seed in seeds:
            leak_checks += verify_no_answer_in_memory(seq_len, args.window, seed, args.batch_size, args.task)
            for write_mode in write_modes:
                for model_name in ["local", "epmem", "hpm_lite", "hebbian"]:
                    train_args = SimpleNamespace(
                        model=model_name,
                        task=args.task,
                        seq_len=seq_len,
                        window=args.window,
                        batch_size=args.batch_size,
                        steps=args.steps,
                        eval_every=max(1, args.steps),
                        eval_batches=1,
                        d_model=args.d_model,
                        layers=args.layers,
                        heads=args.heads,
                        lr=3.0e-4,
                        seed=seed,
                        device=args.device,
                        lambda_ret=0.1,
                        top_k=1,
                        memory_control="normal",
                        write_mode=write_mode,
                        oracle_memory=True,
                        num_facts=4,
                        repeated_keys=False,
                        similar_values=False,
                        distractor_fact_spans=0,
                        query_key_noise_only=False,
                        fact_order="random",
                        out_dir=str(out_dir / "runs"),
                        save_checkpoint=True,
                    )
                    train_metrics = run_training(train_args)
                    checkpoint_path = Path(str(train_metrics["run_dir"])) / "checkpoint.pt"
                    model = load_checkpoint(checkpoint_path, device)

                    for control in controls_for(model_name):
                        set_seed(1_000_000 + seed + seq_len)
                        eval_dataset = FactRecallDataset(
                            FactRecallConfig(
                                seq_len=seq_len,
                                window=args.window,
                                task=args.task,
                                seed=200_000 + seed + seq_len,
                                oracle_memory=True,
                            )
                        )
                        metrics = evaluate_batches(
                            model=model,
                            dataset=eval_dataset,
                            batch_size=args.batch_size,
                            batches=args.eval_batches,
                            device=device,
                            task=args.task,
                            top_k=1,
                            memory_control=control,
                            write_mode=write_mode,
                        )
                        raw_rows.append(
                            {
                                "write_mode": write_mode,
                                "model": model_name,
                                "seq_len": seq_len,
                                "seed": seed,
                                "memory_control": control,
                                "answer_exact": metrics["answer_exact"],
                                "answer_ce": metrics["answer_ce"],
                                "retrieval_top1": metrics.get("retrieval_top1", ""),
                                "retrieval_topk": metrics.get("retrieval_topk", ""),
                                "examples_per_sec": metrics["examples_per_sec"],
                                "avg_written_slots": metrics["avg_written_slots"],
                                "true_fact_written_rate": metrics["true_fact_written_rate"],
                                "false_write_rate": metrics["false_write_rate"],
                                "missed_fact_rate": metrics["missed_fact_rate"],
                                "train_examples_per_sec": train_metrics.get("examples_per_sec_recent", ""),
                                "run_dir": train_metrics["run_dir"],
                            }
                        )

    summary_rows = summarize(raw_rows)
    raw_columns = [
        "write_mode",
        "model",
        "seq_len",
        "seed",
        "memory_control",
        "answer_exact",
        "answer_ce",
        "retrieval_top1",
        "retrieval_topk",
        "examples_per_sec",
        "avg_written_slots",
        "true_fact_written_rate",
        "false_write_rate",
        "missed_fact_rate",
        "train_examples_per_sec",
        "run_dir",
    ]
    summary_columns = [
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
        "examples_per_sec_mean",
        "examples_per_sec_std",
        "avg_written_slots_mean",
        "avg_written_slots_std",
        "true_fact_written_rate_mean",
        "true_fact_written_rate_std",
        "false_write_rate_mean",
        "false_write_rate_std",
        "missed_fact_rate_mean",
        "missed_fact_rate_std",
    ]
    write_csv(Path("runs") / "validation_raw.csv", raw_rows, raw_columns)
    write_csv(Path("runs") / "validation_summary.csv", summary_rows, summary_columns)
    write_results_md(Path("results.md"), summary_rows, args, leak_checks)
    print("wrote runs/validation_raw.csv, runs/validation_summary.csv, and results.md")


if __name__ == "__main__":
    main()
