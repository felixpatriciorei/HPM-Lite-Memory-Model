from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpm_lite.data import FACT, FactRecallConfig, FactRecallDataset
from hpm_lite.evaluate import evaluate_batches
from hpm_lite.metrics import answer_cross_entropy, answer_exact_accuracy, retrieval_metrics
from hpm_lite.model import HpmLiteConfig, HpmLiteModel
from hpm_lite.train import TinyAdamW
from hpm_lite.utils import ensure_dir, resolve_device, set_seed


MODELS = ["local", "epmem", "hpm_lite", "hebbian"]
CONTROLS = ["normal", "corrupt_values", "random_order", "query_key_noise_only"]


@dataclass
class ControlSpec:
    name: str
    memory_control: str = "normal"
    fact_order: str = "query_last"
    query_key_noise_only: bool = False


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Hebbian memory for leakage and near-oracle shortcuts.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--fact-counts", type=str, default="8,16,32")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="runs/hebbian_audit")
    return parser


def hard_config(
    *,
    seq_len: int,
    window: int,
    num_facts: int,
    seed: int,
    fact_order: str,
    query_key_noise_only: bool = False,
) -> FactRecallConfig:
    return FactRecallConfig(
        seq_len=seq_len,
        window=window,
        task="twohop",
        num_facts=num_facts,
        seed=seed,
        oracle_memory=True,
        repeated_keys=True,
        similar_values=True,
        distractor_fact_spans=max(4, num_facts // 4),
        query_key_noise_only=query_key_noise_only,
        fact_order=fact_order,
    )


def audit_memory_batch(batch: Dict[str, torch.Tensor], expect_no_match: bool = False) -> int:
    input_ids = batch["input_ids"]
    answer_positions = batch["answer_positions"]
    query_positions = answer_positions - 2
    answer_value_positions = answer_positions + 1
    memory_spans = batch["memory_spans"]
    memory_positions = batch["memory_token_positions"]
    bsz, slots, _ = memory_positions.shape

    if not torch.all(memory_spans[:, :, 1] < query_positions[:, None]):
        raise AssertionError("memory span reaches the query/answer region")
    if torch.any(memory_positions >= query_positions[:, None, None]):
        raise AssertionError("Hebbian/episodic memory received a non-pre-query token")
    if torch.any(memory_positions == answer_value_positions[:, None, None]):
        raise AssertionError("future answer token leaked into memory positions")

    for b in range(bsz):
        for slot in range(slots):
            start = int(memory_spans[b, slot, 0].item())
            if int(input_ids[b, start].item()) != FACT:
                raise AssertionError("memory span does not start at a FACT token")
        if expect_no_match:
            query_key = int(batch["query_key_tokens"][b].item())
            fact_key_positions = memory_positions[b, :, 0]
            fact_keys = input_ids[b, fact_key_positions]
            if torch.any(fact_keys == query_key):
                raise AssertionError("query_key_noise_only control accidentally stored the query key as a fact")
            if not torch.any(input_ids[b, : int(query_positions[b].item())] == query_key):
                raise AssertionError("query_key_noise_only control did not place query key in pre-query noise")
    return bsz


def train_one_model(
    model_name: str,
    config: FactRecallConfig,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    d_model: int,
    layers: int,
    heads: int,
    device: torch.device,
) -> tuple[HpmLiteModel, Dict[str, float]]:
    set_seed(seed)
    dataset = FactRecallDataset(config)
    model = HpmLiteModel(
        HpmLiteConfig(
            model_type=model_name,
            d_model=d_model,
            layers=layers,
            heads=heads,
            window=config.window,
            max_seq_len=max(2048, config.seq_len + 1),
        )
    ).to(device)
    optimizer = TinyAdamW(model.parameters(), lr=3.0e-4)

    start = time.perf_counter()
    final_loss = 0.0
    final_exact = 0.0
    final_ret: Dict[str, float] = {}
    for _ in range(steps):
        model.train()
        batch = dataset.sample_batch(batch_size, device=device)
        audit_memory_batch(batch)
        output = model(
            batch["input_ids"],
            memory_token_positions=batch["memory_token_positions"],
            memory_mask=batch["memory_mask"],
            answer_positions=batch["answer_positions"],
            query_key_positions=batch["query_key_positions"],
            task=config.task,
            top_k=1,
            hop_positive_memory_indices=batch["hop_positive_memory_indices"],
        )
        logits = output["logits"]
        answer_loss = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        retrieval_loss = output["retrieval"].get("retrieval_loss", logits.new_zeros(()))
        loss = answer_loss + 0.1 * retrieval_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for {model_name}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())
        final_exact = float(answer_exact_accuracy(logits.detach(), batch["answer_positions"], batch["answer_tokens"]).item())
        final_ret = retrieval_metrics(output["retrieval"], batch["positive_memory_indices"])

    elapsed = time.perf_counter() - start
    return model, {
        "train_loss": final_loss,
        "train_exact": final_exact,
        "train_retrieval_top1": final_ret.get("retrieval_top1", ""),
        "train_examples_per_sec": (steps * batch_size) / max(elapsed, 1.0e-9),
    }


def control_specs() -> List[ControlSpec]:
    return [
        ControlSpec("normal"),
        ControlSpec("corrupt_values", memory_control="corrupt_values"),
        ControlSpec("random_order", fact_order="random"),
        ControlSpec("query_key_noise_only", query_key_noise_only=True),
    ]


def audit_eval_dataset(dataset: FactRecallDataset, batch_size: int, batches: int, expect_no_match: bool) -> int:
    checked = 0
    for _ in range(batches):
        batch = dataset.sample_batch(batch_size)
        checked += audit_memory_batch(batch, expect_no_match=expect_no_match)
    return checked


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[int, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (int(row["num_facts"]), str(row["model"]), str(row["control"]))
        grouped.setdefault(key, []).append(row)

    summary: List[Dict[str, object]] = []
    metrics = ["answer_exact", "answer_ce", "retrieval_top1", "examples_per_sec"]
    for key in sorted(grouped):
        num_facts, model, control = key
        group = grouped[key]
        out: Dict[str, object] = {
            "num_facts": num_facts,
            "model": model,
            "control": control,
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
        summary.append(out)
    return summary


def mean_for(summary: List[Dict[str, object]], model: str, control: str) -> float | None:
    values = [
        float(row["answer_exact_mean"])
        for row in summary
        if row["model"] == model and row["control"] == control and row["answer_exact_mean"] != ""
    ]
    return statistics.mean(values) if values else None


def verdict(summary: List[Dict[str, object]]) -> str:
    hebb_normal = mean_for(summary, "hebbian", "normal")
    hebb_corrupt = mean_for(summary, "hebbian", "corrupt_values")
    hebb_random = mean_for(summary, "hebbian", "random_order")
    hebb_nomatch = mean_for(summary, "hebbian", "query_key_noise_only")
    epmem_normal = mean_for(summary, "epmem", "normal")

    pieces = []
    if hebb_normal is not None and epmem_normal is not None:
        pieces.append(
            f"Hebbian normal exact averaged {hebb_normal:.3f}, versus slot epmem at {epmem_normal:.3f}."
        )
        pieces.append(
            "The slot-style retrieval top1 column is not a faithful explanation for Hebbian success under repeated keys; "
            "the dense matrix can output the right value through recency-weighted superposition even when the labeled slot is not top1."
        )
    if hebb_normal is not None and hebb_random is not None:
        pieces.append(
            f"Randomizing fact order changed Hebbian by {(hebb_random - hebb_normal) * 100:.1f} percentage points."
        )
    if hebb_normal is not None and hebb_corrupt is not None:
        pieces.append(
            f"Partially corrupting values changed Hebbian by {(hebb_corrupt - hebb_normal) * 100:.1f} percentage points."
        )
    if hebb_nomatch is not None:
        pieces.append(f"When the query key appeared only in noise, Hebbian exact averaged {hebb_nomatch:.3f}.")

    if hebb_normal is not None and hebb_corrupt is not None and hebb_random is not None:
        if hebb_normal > 0.8 and (hebb_corrupt < hebb_normal - 0.2 or hebb_random < hebb_normal - 0.2):
            pieces.append(
                "Verdict: Hebbian is exploiting the ordered oracle/token setup; the controls expose brittleness."
            )
        elif hebb_normal > 0.8:
            pieces.append(
                "Verdict: Hebbian stayed strong under these controls, but oracle writes and tied token embeddings remain a major shortcut."
            )
        else:
            pieces.append("Verdict: Hebbian is not genuinely robust in this harder setting.")
    return " ".join(pieces)


def write_results_md(
    path: Path,
    summary: List[Dict[str, object]],
    args: argparse.Namespace,
    leak_checks: int,
) -> None:
    lines = [
        "# HPM-Lite Hebbian Audit",
        "",
        "## What Was Run",
        "",
        f"Task: `twohop`; facts per sequence: `{args.fact_counts}`; seeds: `{args.seeds}`.",
        f"Hard data: repeated/confusable keys, adjacent value IDs, and `{max(4, max(parse_int_list(args.fact_counts)) // 4)}` FACT-like distractor spans at the largest fact count.",
        f"Sequence length/window: `{args.seq_len}` / `{args.window}`.",
        f"Training: `{args.steps}` steps, batch size `{args.batch_size}`; eval batches `{args.eval_batches}`.",
        f"Device request: `{args.device}`. Platform: `{platform.platform()}`. Torch: `{torch.__version__}`.",
        f"Leak audit passed on `{leak_checks}` generated examples: memory positions were pre-query FACT spans and never included the post-ANSWER token.",
        "",
        "Command:",
        "",
        f"- `python scripts/run_hebbian_audit.py --steps {args.steps} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --d-model {args.d_model} --layers {args.layers} --heads {args.heads} --device {args.device}`",
        "",
        "Controls:",
        "",
        "- `normal`: query chain facts are stored after distractors.",
        "- `corrupt_values`: keys are unchanged but every other stored value is replaced by another memory value.",
        "- `random_order`: the same hard generator stores facts in random order.",
        "- `query_key_noise_only`: the query key appears before the query as noise but is not stored as a fact.",
        "",
        "## Summary Table",
        "",
        "| facts | model | control | exact mean | exact std | CE mean | CE std | ret top1 mean | examples/sec mean |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        ret = row["retrieval_top1_mean"]
        ret_text = "" if ret == "" else f"{float(ret):.4f}"
        lines.append(
            "| {facts} | {model} | {control} | {exact:.4f} | {exact_std:.4f} | {ce:.4f} | {ce_std:.4f} | {ret} | {eps:.2f} |".format(
                facts=row["num_facts"],
                model=row["model"],
                control=row["control"],
                exact=float(row["answer_exact_mean"]),
                exact_std=float(row["answer_exact_std"]),
                ce=float(row["answer_ce_mean"]),
                ce_std=float(row["answer_ce_std"]),
                ret=ret_text,
                eps=float(row["examples_per_sec_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict(summary),
            "",
            "This audit is still CPU-small. It is designed to catch leakage and toy shortcuts, not to settle model quality.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    seeds = parse_int_list(args.seeds)
    fact_counts = parse_int_list(args.fact_counts)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    rows: List[Dict[str, object]] = []
    leak_checks = 0
    specs = control_specs()
    for num_facts in fact_counts:
        for seed in seeds:
            train_config = hard_config(
                seq_len=args.seq_len,
                window=args.window,
                num_facts=num_facts,
                seed=seed,
                fact_order="query_last",
            )
            for model_name in MODELS:
                model, train_metrics = train_one_model(
                    model_name,
                    train_config,
                    seed=seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    d_model=args.d_model,
                    layers=args.layers,
                    heads=args.heads,
                    device=device,
                )
                for control_index, spec in enumerate(specs):
                    eval_seed = 100_000 + seed + 10_000 * num_facts + control_index
                    eval_config = hard_config(
                        seq_len=args.seq_len,
                        window=args.window,
                        num_facts=num_facts,
                        seed=eval_seed,
                        fact_order=spec.fact_order,
                        query_key_noise_only=spec.query_key_noise_only,
                    )
                    audit_dataset = FactRecallDataset(eval_config)
                    leak_checks += audit_eval_dataset(
                        audit_dataset,
                        batch_size=args.batch_size,
                        batches=1,
                        expect_no_match=spec.query_key_noise_only,
                    )
                    eval_dataset = FactRecallDataset(eval_config)
                    metrics = evaluate_batches(
                        model=model,
                        dataset=eval_dataset,
                        batch_size=args.batch_size,
                        batches=args.eval_batches,
                        device=device,
                        task="twohop",
                        top_k=1,
                        memory_control=spec.memory_control,
                    )
                    rows.append(
                        {
                            "num_facts": num_facts,
                            "seed": seed,
                            "model": model_name,
                            "control": spec.name,
                            "answer_exact": metrics["answer_exact"],
                            "answer_ce": metrics["answer_ce"],
                            "retrieval_top1": metrics.get("retrieval_top1", ""),
                            "examples_per_sec": metrics["examples_per_sec"],
                            **train_metrics,
                        }
                    )
                print(f"done facts={num_facts} seed={seed} model={model_name}")

    summary = summarize(rows)
    raw_columns = [
        "num_facts",
        "seed",
        "model",
        "control",
        "answer_exact",
        "answer_ce",
        "retrieval_top1",
        "examples_per_sec",
        "train_loss",
        "train_exact",
        "train_retrieval_top1",
        "train_examples_per_sec",
    ]
    summary_columns = [
        "num_facts",
        "model",
        "control",
        "n",
        "answer_exact_mean",
        "answer_exact_std",
        "answer_ce_mean",
        "answer_ce_std",
        "retrieval_top1_mean",
        "retrieval_top1_std",
        "examples_per_sec_mean",
        "examples_per_sec_std",
    ]
    write_csv(out_dir / "raw.csv", rows, raw_columns)
    write_csv(out_dir / "summary.csv", summary, summary_columns)
    write_results_md(Path("results.md"), summary, args, leak_checks)
    print(f"wrote {out_dir / 'raw.csv'}, {out_dir / 'summary.csv'}, and results.md")


if __name__ == "__main__":
    main()
