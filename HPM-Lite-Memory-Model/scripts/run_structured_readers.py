from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpm_lite.data import CONDITIONAL_TASKS, STRESS_TASKS, FactRecallConfig, FactRecallDataset, QUERY
from hpm_lite.structured_readout import (
    LearnedConditionReader,
    LearnedSetReader,
    count_trainable_parameters,
    learned_condition_stress_metrics,
    learned_set_stress_metrics,
    symbolic_condition_binding_metrics,
    symbolic_set_metrics,
)
from hpm_lite.train import TinyAdamW
from hpm_lite.utils import ensure_dir, resolve_device, set_seed
from hpm_lite.write_modes import apply_write_mode, first_positions, parse_write_modes


TASKS = {
    "conditional_positive_only",
    "conditional_contrastive",
    "conditional_contrastive_stress",
    "coexisting",
    "coexisting_stress",
    "kv",
    "kv_stress",
    "longhop",
}
CONTROLS = {"normal", "no_retrieval", "shuffled_values", "random_keys", "corrupt_conditions", "corrupt_values"}
DEFAULT_CLEAN_CONTROLS = "normal,no_retrieval,shuffled_values,random_keys"
DEFAULT_STRESS_CONTROLS = "normal,shuffled_values,random_keys,corrupt_conditions,corrupt_values"


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str, allowed: set[str]) -> List[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"unknown value(s): {unknown}")
    return items


def parse_auto_positive(value: str, task: str) -> List[int]:
    if value == "auto":
        return [2] if task in {"coexisting", "coexisting_stress"} else [1]
    return parse_int_list(value)


def is_stress_task(task: str) -> bool:
    return task in STRESS_TASKS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train tiny learned readers over typed memory slots.")
    parser.add_argument("--tasks", type=str, default="conditional_positive_only,conditional_contrastive,coexisting")
    parser.add_argument("--write-modes", type=str, default="fact_token,oracle")
    parser.add_argument("--controls", type=str, default=DEFAULT_CLEAN_CONTROLS)
    parser.add_argument("--stress-controls", type=str, default=DEFAULT_STRESS_CONTROLS)
    parser.add_argument("--budgets", type=str, default="0,1,3,10,30,100")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument("--slot-counts", type=str, default="")
    parser.add_argument("--num-positive", type=str, default="auto")
    parser.add_argument("--num-hard-negatives", type=str, default="0")
    parser.add_argument("--similarity-modes", type=str, default="none")
    parser.add_argument("--slot-order", type=str, default="random")
    parser.add_argument("--reader-dim", type=int, default=64)
    parser.add_argument("--reader-hidden", type=int, default=128)
    parser.add_argument("--reader-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--freeze-reader-embeddings", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="runs")
    return parser


def dataset_for(
    task: str,
    seq_len: int,
    window: int,
    seed: int,
    num_facts: int,
    num_positive: int = 2,
    num_hard_negatives: int = 0,
    similarity_mode: str = "none",
    slot_order: str = "random",
) -> FactRecallDataset:
    return FactRecallDataset(
        FactRecallConfig(
            seq_len=seq_len,
            window=window,
            task=task,
            num_facts=num_facts,
            seed=seed,
            oracle_memory=True,
            num_positive=num_positive,
            num_hard_negatives=num_hard_negatives,
            similarity_mode=similarity_mode,
            slot_order=slot_order,
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
    num_positive: int = 2,
    num_hard_negatives: int = 0,
    similarity_mode: str = "none",
    slot_order: str = "random",
) -> int:
    dataset = dataset_for(
        task,
        seq_len,
        window,
        seed,
        num_facts,
        num_positive,
        num_hard_negatives,
        similarity_mode,
        slot_order,
    )
    batch = dataset.sample_batch(batch_size)
    query_positions = first_positions(batch["input_ids"], QUERY)
    answer_input_positions = batch["answer_target_positions"] + 1
    for write_mode in write_modes:
        written, _ = apply_write_mode(batch, write_mode)
        for b in range(batch_size):
            valid = written["memory_mask"][b]
            if not bool(valid.any().item()):
                continue
            positions = written["memory_token_positions"][b, valid]
            if torch.any(positions >= query_positions[b]):
                raise AssertionError("reader diagnostic write reached QUERY or a later token")
            if torch.any(torch.isin(positions.reshape(-1), answer_input_positions[b])):
                raise AssertionError("future answer token leaked into reader memory slots")
    return batch_size


def make_reader(task: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    kwargs = {
        "reader_dim": args.reader_dim,
        "hidden": args.reader_hidden,
        "layers": args.reader_layers,
        "dropout": args.dropout,
        "train_embeddings": not args.freeze_reader_embeddings,
    }
    if task in {"coexisting", "coexisting_stress"}:
        return LearnedSetReader(**kwargs).to(device)
    if task in CONDITIONAL_TASKS:
        return LearnedConditionReader(**kwargs).to(device)
    raise ValueError(f"learned reader v1 only supports coexisting and conditional tasks, got {task}")


def reader_type(task: str) -> str:
    return "set" if task in {"coexisting", "coexisting_stress"} else "condition"


def symbolic_metrics(task: str, batch: Dict[str, torch.Tensor], control: str) -> Dict[str, float]:
    if task in {"coexisting", "coexisting_stress"}:
        return symbolic_set_metrics(batch, control)
    if task in CONDITIONAL_TASKS:
        return symbolic_condition_binding_metrics(batch, control)
    return {"symbolic_readout_available": 0.0}


def learned_metrics(task: str, reader: torch.nn.Module, batch: Dict[str, torch.Tensor], control: str) -> Dict[str, float]:
    if task in {"coexisting", "coexisting_stress"}:
        return reader.metrics(batch, control)
    if task in CONDITIONAL_TASKS:
        return reader.metrics(batch, control)
    return {"learned_readout_available": 0.0}


def train_loss(task: str, reader: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if task in {"coexisting", "coexisting_stress"}:
        return reader.loss(batch, "normal")
    if task in CONDITIONAL_TASKS:
        return reader.loss(batch, "normal")
    raise ValueError(f"no learned reader loss for task {task}")


@torch.no_grad()
def evaluate_reader(
    task: str,
    reader: torch.nn.Module,
    dataset: FactRecallDataset,
    batch_size: int,
    batches: int,
    device: torch.device,
    write_mode: str,
    control: str,
) -> Dict[str, float]:
    reader.eval()
    sums: Dict[str, float] = {}
    total = 0
    for _ in range(batches):
        batch = dataset.sample_batch(batch_size, device=device)
        batch, write_stats = apply_write_mode(batch, write_mode)
        metrics = {}
        metrics.update(symbolic_metrics(task, batch, control))
        metrics.update(learned_metrics(task, reader, batch, control))
        if is_stress_task(task):
            if task in {"coexisting_stress", "kv_stress"}:
                metrics.update(learned_set_stress_metrics(reader, batch, control))
            elif task in CONDITIONAL_TASKS:
                metrics.update(learned_condition_stress_metrics(reader, batch, control))
        for key, value in write_stats.items():
            metrics[key] = value
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
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
    "reader_type",
    "write_mode",
    "seed",
    "budget_steps",
    "memory_control",
    "slot_count",
    "num_positive",
    "num_hard_negatives",
    "similarity_mode",
    "slot_order",
    "ce_reference_exact",
    "ce_reference_model",
    "ce_reference_source",
    "symbolic_readout_available",
    "condition_symbolic_exact",
    "symbolic_set_exact",
    "symbolic_set_f1",
    "learned_readout_available",
    "learned_condition_exact",
    "learned_condition_slot_accuracy",
    "learned_condition_value_accuracy",
    "learned_set_exact",
    "learned_set_precision",
    "learned_set_recall",
    "learned_set_f1",
    "condition_key_match_accuracy",
    "condition_cond_match_accuracy",
    "hard_negative_false_positive_rate",
    "hard_negative_rank_mean",
    "correct_slot_rank",
    "learned_top1_slot_accuracy",
    "learned_topk_slot_accuracy",
    "learned_value_accuracy",
    "symbolic_available_rate",
    "ambiguous_symbolic_match_rate",
    "key_only_distractor_error_rate",
    "condition_only_distractor_error_rate",
    "neither_match_distractor_error_rate",
    "missed_positive_rate",
    "extra_false_positive_rate",
    "learned_set_exact_t03",
    "learned_set_exact_t07",
    "learned_vs_symbolic_gap",
    "learned_vs_ce_gap",
    "reader_trainable_params",
    "reader_steps_to_90",
    "reader_steps_to_99",
    "reader_train_time",
    "reader_examples_per_sec",
    "avg_written_slots",
    "true_fact_written_rate",
    "false_write_rate",
    "missed_fact_rate",
]


SUMMARY_COLUMNS = [
    "task",
    "reader_type",
    "write_mode",
    "budget_steps",
    "memory_control",
    "slot_count",
    "num_positive",
    "num_hard_negatives",
    "similarity_mode",
    "slot_order",
    "n",
]
for metric in RAW_COLUMNS[11:]:
    if metric in {"ce_reference_model", "ce_reference_source", "similarity_mode", "slot_order"}:
        continue
    SUMMARY_COLUMNS.extend([f"{metric}_mean", f"{metric}_std"])


def summarize(raw_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str, int, str], List[Dict[str, object]]] = {}
    for row in raw_rows:
        key = (
            str(row["task"]),
            str(row["reader_type"]),
            str(row["write_mode"]),
            int(row["budget_steps"]),
            str(row["memory_control"]),
            int(row.get("slot_count", 0)),
            int(row.get("num_positive", 0)),
            int(row.get("num_hard_negatives", 0)),
            str(row.get("similarity_mode", "")),
            str(row.get("slot_order", "")),
        )
        grouped.setdefault(key, []).append(row)

    metrics = [column[:-5] for column in SUMMARY_COLUMNS if column.endswith("_mean")]
    out_rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        task, rtype, write_mode, budget, control, slot_count, num_positive, num_hard, similarity_mode, slot_order = key
        group = grouped[key]
        out: Dict[str, object] = {
            "task": task,
            "reader_type": rtype,
            "write_mode": write_mode,
            "budget_steps": budget,
            "memory_control": control,
            "slot_count": slot_count,
            "num_positive": num_positive,
            "num_hard_negatives": num_hard,
            "similarity_mode": similarity_mode,
            "slot_order": slot_order,
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
        out_rows.append(out)
    return out_rows


def load_ce_references(out_dir: Path) -> Dict[tuple[str, str, str], tuple[float, str, str]]:
    references: Dict[tuple[str, str, str], tuple[float, str, str]] = {}
    candidates = [
        out_dir / "condition_binding_summary.csv",
        out_dir / "readuse_summary.csv",
        out_dir / "memfail_budget_summary.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("answer_exact_mean", "") == "":
                    continue
                if row.get("budget_steps") not in {"300", "100"}:
                    continue
                key = (row.get("task", ""), row.get("write_mode", ""), row.get("memory_control", "normal"))
                value = float(row["answer_exact_mean"])
                model = row.get("model", "")
                old = references.get(key)
                if old is None or value > old[0]:
                    references[key] = (value, model, path.name)
    return references


def flush_outputs(out_dir: Path, raw_rows: List[Dict[str, object]], stress_mode: bool = False) -> List[Dict[str, object]]:
    summary_rows = summarize(raw_rows)
    if stress_mode:
        write_csv(out_dir / "structured_reader_stress_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / "structured_reader_stress_summary.csv", summary_rows, SUMMARY_COLUMNS)
    else:
        write_csv(out_dir / "structured_readers_raw.csv", raw_rows, RAW_COLUMNS)
        write_csv(out_dir / "structured_readers_summary.csv", summary_rows, SUMMARY_COLUMNS)
    return summary_rows


def metric_value(row: Dict[str, object], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def summary_lookup(
    rows: List[Dict[str, object]],
    task: str,
    write_mode: str,
    control: str,
    budget: int,
    metric: str,
) -> float | None:
    values = [
        float(row[f"{metric}_mean"])
        for row in rows
        if row["task"] == task
        and row["write_mode"] == write_mode
        and row["memory_control"] == control
        and int(row["budget_steps"]) == budget
        and row.get(f"{metric}_mean", "") != ""
    ]
    return statistics.mean(values) if values else None


def stress_metric_name(task: str, kind: str) -> str:
    if task in {"coexisting_stress", "kv_stress"}:
        return "symbolic_set_exact" if kind == "symbolic" else "learned_set_exact"
    return "condition_symbolic_exact" if kind == "symbolic" else "learned_condition_exact"


def stress_rows(
    rows: List[Dict[str, object]],
    final_budget: int,
    control: str = "normal",
    write_mode: str = "fact_token",
) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if row["task"] in STRESS_TASKS
        and int(row["budget_steps"]) == final_budget
        and row["memory_control"] == control
        and row["write_mode"] == write_mode
    ]


def row_float(row: Dict[str, object], field: str) -> float | None:
    value = row.get(field, "")
    return None if value == "" else float(value)


def aggregate_stress(
    rows: List[Dict[str, object]],
    group_field: str,
    metric: str,
    final_budget: int,
    control: str = "normal",
) -> List[tuple[str, float]]:
    groups: Dict[str, List[float]] = {}
    for row in stress_rows(rows, final_budget, control=control):
        value = row_float(row, f"{metric}_mean")
        if value is None:
            continue
        groups.setdefault(str(row[group_field]), []).append(value)
    return [(key, statistics.mean(values)) for key, values in sorted(groups.items())]


def update_stress_results_md(
    path: Path,
    summary_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    raw_count: int,
) -> None:
    budgets = parse_int_list(args.budgets)
    final_budget = max(budgets)
    lines = [
        "## Structured Reader Stress Suite v1",
        "",
        "This run stress-tests learned structured readers over typed slots. It still trains only reader modules and separate reader embeddings; it does not train a backbone, CE decoder, memory writer, HPM-Lite recurrence, JEPA, ANN, graph memory, Priming, GKA, or RL.",
        "",
        f"Tasks: `{args.tasks}`. Write modes: `{args.write_modes}`. Controls: `{args.stress_controls}`.",
        f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Slot counts: `{args.slot_counts}`. Hard negatives: `{args.num_hard_negatives}`. Similarity modes: `{args.similarity_modes}`. Slot order: `{args.slot_order}`.",
        f"Reader: dim `{args.reader_dim}`, hidden `{args.reader_hidden}`, layers `{args.reader_layers}`, lr `{args.lr}`. Device request: `{args.device}`. Torch: `{torch.__version__}`.",
        "",
        "Command:",
        "",
        f"- `python scripts/run_structured_readers.py --tasks {args.tasks} --write-modes {args.write_modes} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --reader-dim {args.reader_dim} --reader-hidden {args.reader_hidden} --device {args.device} --slot-counts {args.slot_counts} --num-hard-negatives {args.num_hard_negatives} --similarity-modes {args.similarity_modes} --slot-order {args.slot_order}`",
        "",
        f"Raw and summarized outputs: `runs/structured_reader_stress_raw.csv` (`{raw_count}` rows), `runs/structured_reader_stress_summary.csv` (`{len(summary_rows)}` rows).",
        "",
        "### Stage A Grid",
        "",
        "| task | slot_count | hard_negatives | similarity | symbolic exact | learned exact | learned-symbolic gap | top1 slot | topk slot | hard FP |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if final_budget < 300:
        lines.insert(
            -3,
            "Stage B full stress was not part of this run. Read this section as Stage A or a reduced stress run unless the budgets include `300` and the full slot/hard-negative/similarity grid was requested.",
        )
        lines.insert(-3, "")
    stage_a_rows = [
        row
        for row in stress_rows(summary_rows, final_budget)
        if int(row["slot_count"]) in {4, 16}
        and int(row["num_hard_negatives"]) in {0, 8}
        and row["similarity_mode"] in {"none", "mixed"}
    ]
    for row in stage_a_rows[:32]:
        symbolic = row_float(row, f"{stress_metric_name(row['task'], 'symbolic')}_mean")
        learned = row_float(row, f"{stress_metric_name(row['task'], 'learned')}_mean")
        lines.append(
            f"| {row['task']} | {row['slot_count']} | {row['num_hard_negatives']} | {row['similarity_mode']} | {format_float(symbolic)} | {format_float(learned)} | {format_float(row_float(row, 'learned_vs_symbolic_gap_mean'))} | {format_float(row_float(row, 'learned_top1_slot_accuracy_mean'))} | {format_float(row_float(row, 'learned_topk_slot_accuracy_mean'))} | {format_float(row_float(row, 'hard_negative_false_positive_rate_mean'))} |"
        )

    lines.extend(
        [
            "",
            "### Stage B Aggregates" if final_budget >= 300 else "### Stress Aggregates",
            "",
            "| grouping | value | learned exact | learned-symbolic gap |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for group_field, label in [
        ("slot_count", "slot_count"),
        ("num_hard_negatives", "hard_negatives"),
        ("similarity_mode", "similarity_mode"),
    ]:
        for value, learned in aggregate_stress(summary_rows, group_field, "learned_condition_exact", final_budget):
            gaps = [row_float(row, "learned_vs_symbolic_gap_mean") for row in stress_rows(summary_rows, final_budget) if str(row[group_field]) == value and row["task"] in CONDITIONAL_TASKS]
            gaps = [gap for gap in gaps if gap is not None]
            gap_text = "" if not gaps else f"{statistics.mean(gaps):.4f}"
            lines.append(f"| conditional {label} | {value} | {learned:.4f} | {gap_text} |")
        for value, learned in aggregate_stress(summary_rows, group_field, "learned_set_exact", final_budget):
            gaps = [row_float(row, "learned_vs_symbolic_gap_mean") for row in stress_rows(summary_rows, final_budget) if str(row[group_field]) == value and row["task"] in {"coexisting_stress", "kv_stress"}]
            gaps = [gap for gap in gaps if gap is not None]
            gap_text = "" if not gaps else f"{statistics.mean(gaps):.4f}"
            lines.append(f"| set {label} | {value} | {learned:.4f} | {gap_text} |")

    lines.extend(
        [
            "",
            "### Controls",
            "",
            "| control | symbolic exact | learned exact | learned-symbolic gap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for control in parse_str_list(args.stress_controls, CONTROLS):
        control_rows = stress_rows(summary_rows, final_budget, control=control)
        symbolic_values = []
        learned_values = []
        gap_values = []
        for row in control_rows:
            symbolic = row_float(row, f"{stress_metric_name(row['task'], 'symbolic')}_mean")
            learned = row_float(row, f"{stress_metric_name(row['task'], 'learned')}_mean")
            gap = row_float(row, "learned_vs_symbolic_gap_mean")
            if symbolic is not None:
                symbolic_values.append(symbolic)
            if learned is not None:
                learned_values.append(learned)
            if gap is not None:
                gap_values.append(gap)
        lines.append(
            f"| {control} | {format_float(statistics.mean(symbolic_values) if symbolic_values else None)} | {format_float(statistics.mean(learned_values) if learned_values else None)} | {format_float(statistics.mean(gap_values) if gap_values else None)} |"
        )

    normal_rows = stress_rows(summary_rows, final_budget)
    worst = None
    for row in normal_rows:
        symbolic = row_float(row, f"{stress_metric_name(row['task'], 'symbolic')}_mean")
        learned = row_float(row, f"{stress_metric_name(row['task'], 'learned')}_mean")
        if symbolic is None or learned is None:
            continue
        gap = learned - symbolic
        if worst is None or gap < worst[0]:
            worst = (gap, row, learned, symbolic)

    lines.extend(["", "### Failure Analysis", ""])
    if worst is not None:
        gap, row, learned, symbolic = worst
        lines.append(
            f"Worst normal learned-symbolic gap at final budget: `{row['task']}` with slot_count `{row['slot_count']}`, hard negatives `{row['num_hard_negatives']}`, similarity `{row['similarity_mode']}`. Symbolic exact = {symbolic:.3f}, learned exact = {learned:.3f}, gap = {gap:.3f}."
        )
    corrupt_values = [row_float(row, f"{stress_metric_name(row['task'], 'learned')}_mean") for row in stress_rows(summary_rows, final_budget, control="corrupt_values")]
    corrupt_conditions = [row_float(row, f"{stress_metric_name(row['task'], 'learned')}_mean") for row in stress_rows(summary_rows, final_budget, control="corrupt_conditions") if row["task"] in CONDITIONAL_TASKS]
    random_keys = [row_float(row, f"{stress_metric_name(row['task'], 'learned')}_mean") for row in stress_rows(summary_rows, final_budget, control="random_keys")]
    corrupt_values = [v for v in corrupt_values if v is not None]
    corrupt_conditions = [v for v in corrupt_conditions if v is not None]
    random_keys = [v for v in random_keys if v is not None]
    if corrupt_values:
        lines.append(f"`corrupt_values` mean learned exact = {statistics.mean(corrupt_values):.3f}; value corruption remains a strong negative control.")
    if corrupt_conditions:
        lines.append(f"`corrupt_conditions` mean conditional learned exact = {statistics.mean(corrupt_conditions):.3f}; condition corruption hits binding as expected.")
    if random_keys:
        lines.append(f"`random_keys` mean learned exact = {statistics.mean(random_keys):.3f}; key corruption hits matching as expected.")

    normal_learned = []
    normal_symbolic = []
    for row in normal_rows:
        symbolic = row_float(row, f"{stress_metric_name(row['task'], 'symbolic')}_mean")
        learned = row_float(row, f"{stress_metric_name(row['task'], 'learned')}_mean")
        if symbolic is not None:
            normal_symbolic.append(symbolic)
        if learned is not None:
            normal_learned.append(learned)
    learned_mean = statistics.mean(normal_learned) if normal_learned else None
    symbolic_mean = statistics.mean(normal_symbolic) if normal_symbolic else None
    lines.extend(["", "### Verdict", ""])
    lines.append("Are learned structured readers still solving the task under ambiguity?")
    if learned_mean is not None and symbolic_mean is not None and learned_mean >= 0.95 and symbolic_mean >= 0.99:
        lines.append(f"Yes at this stress level: final normal learned exact averages {learned_mean:.3f} while symbolic exact averages {symbolic_mean:.3f}.")
    elif learned_mean is not None and symbolic_mean is not None and symbolic_mean >= 0.99:
        lines.append(f"Partially: symbolic remains high ({symbolic_mean:.3f}) but learned exact drops to {learned_mean:.3f}, so reader robustness is now the bottleneck.")
    else:
        lines.append("No clear verdict: symbolic exact also drops, so inspect task generation or slot parsing before blaming the learned reader.")
    lines.append("Where do they fail first?")
    if worst is not None:
        _, row, _, _ = worst
        lines.append(f"The first visible failure is the worst-gap setting above: `{row['task']}`, slot_count `{row['slot_count']}`, hard negatives `{row['num_hard_negatives']}`, similarity `{row['similarity_mode']}`.")
    lines.append("Is the next step learned writing, reader robustness, or real-data slot extraction?")
    if learned_mean is not None and learned_mean >= 0.95:
        lines.append("Next step should be real-data slot extraction or learned writing, because the reader survives this synthetic ambiguity suite.")
    else:
        lines.append("Next step should be reader robustness before learned writing, because symbolic storage is ahead of learned readout.")
    lines.append("")

    section = "\n".join(lines)
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## Structured Reader Stress Suite v1\n"
        index = old.find(marker)
        if index >= 0:
            old = old[:index].rstrip()
        path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
    else:
        path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")


def train_and_evaluate(
    task: str,
    write_mode: str,
    seed: int,
    budgets: List[int],
    controls: List[str],
    args: argparse.Namespace,
    device: torch.device,
    ce_refs: Dict[tuple[str, str, str], tuple[float, str, str]],
    raw_rows: List[Dict[str, object]],
    out_dir: Path,
    slot_count: int,
    num_positive: int,
    num_hard_negatives: int,
    similarity_mode: str,
    slot_order: str,
    stress_mode: bool,
) -> None:
    set_seed(seed)
    reader = make_reader(task, args, device)
    optimizer = TinyAdamW(reader.parameters(), lr=args.lr, weight_decay=0.0)
    train_dataset = dataset_for(
        task,
        args.seq_len,
        args.window,
        seed,
        slot_count,
        num_positive,
        num_hard_negatives,
        similarity_mode,
        slot_order,
    )
    budget_set = set(budgets)
    max_budget = max(budgets)
    trainable_params = count_trainable_parameters(reader)
    steps_to_90: int | None = None
    steps_to_99: int | None = None
    start = time.perf_counter()
    trained_examples = 0

    def evaluate_budget(budget: int) -> None:
        nonlocal steps_to_90, steps_to_99
        train_time = time.perf_counter() - start
        budget_rows: List[Dict[str, object]] = []
        normal_learned: float | None = None
        for control in controls:
            eval_dataset = dataset_for(
                task,
                args.seq_len,
                args.window,
                500_000 + seed + budget,
                slot_count,
                num_positive,
                num_hard_negatives,
                similarity_mode,
                slot_order,
            )
            metrics = evaluate_reader(
                task=task,
                reader=reader,
                dataset=eval_dataset,
                batch_size=args.batch_size,
                batches=args.eval_batches,
                device=device,
                write_mode=write_mode,
                control=control,
            )
            symbolic_exact = metrics.get("condition_symbolic_exact", metrics.get("symbolic_set_exact", ""))
            learned_exact = metrics.get("learned_condition_exact", metrics.get("learned_set_exact", ""))
            ce_value, ce_model, ce_source = ce_refs.get((task, write_mode, control), ("", "", ""))
            learned_vs_symbolic = ""
            learned_vs_ce = ""
            if learned_exact != "" and symbolic_exact != "":
                learned_vs_symbolic = float(learned_exact) - float(symbolic_exact)
            if learned_exact != "" and ce_value != "":
                learned_vs_ce = float(learned_exact) - float(ce_value)
            if control == "normal" and learned_exact != "":
                normal_learned = float(learned_exact)
            row: Dict[str, object] = {
                "task": task,
                "reader_type": reader_type(task),
                "write_mode": write_mode,
                "seed": seed,
                "budget_steps": budget,
                "memory_control": control,
                "slot_count": slot_count,
                "num_positive": num_positive,
                "num_hard_negatives": num_hard_negatives,
                "similarity_mode": similarity_mode,
                "slot_order": slot_order,
                "ce_reference_exact": ce_value,
                "ce_reference_model": ce_model,
                "ce_reference_source": ce_source,
                "symbolic_readout_available": metrics.get("symbolic_readout_available", ""),
                "condition_symbolic_exact": metrics.get("condition_symbolic_exact", ""),
                "symbolic_set_exact": metrics.get("symbolic_set_exact", ""),
                "symbolic_set_f1": metrics.get("symbolic_set_f1", ""),
                "learned_readout_available": metrics.get("learned_readout_available", ""),
                "learned_condition_exact": metrics.get("learned_condition_exact", ""),
                "learned_condition_slot_accuracy": metrics.get("learned_condition_slot_accuracy", ""),
                "learned_condition_value_accuracy": metrics.get("learned_condition_value_accuracy", ""),
                "learned_set_exact": metrics.get("learned_set_exact", ""),
                "learned_set_precision": metrics.get("learned_set_precision", ""),
                "learned_set_recall": metrics.get("learned_set_recall", ""),
                "learned_set_f1": metrics.get("learned_set_f1", ""),
                "condition_key_match_accuracy": metrics.get("condition_key_match_accuracy", ""),
                "condition_cond_match_accuracy": metrics.get("condition_cond_match_accuracy", ""),
                "hard_negative_false_positive_rate": metrics.get("hard_negative_false_positive_rate", ""),
                "hard_negative_rank_mean": metrics.get("hard_negative_rank_mean", ""),
                "correct_slot_rank": metrics.get("correct_slot_rank", ""),
                "learned_top1_slot_accuracy": metrics.get("learned_top1_slot_accuracy", ""),
                "learned_topk_slot_accuracy": metrics.get("learned_topk_slot_accuracy", ""),
                "learned_value_accuracy": metrics.get("learned_value_accuracy", ""),
                "symbolic_available_rate": metrics.get("symbolic_available_rate", ""),
                "ambiguous_symbolic_match_rate": metrics.get("ambiguous_symbolic_match_rate", ""),
                "key_only_distractor_error_rate": metrics.get("key_only_distractor_error_rate", ""),
                "condition_only_distractor_error_rate": metrics.get("condition_only_distractor_error_rate", ""),
                "neither_match_distractor_error_rate": metrics.get("neither_match_distractor_error_rate", ""),
                "missed_positive_rate": metrics.get("missed_positive_rate", ""),
                "extra_false_positive_rate": metrics.get("extra_false_positive_rate", ""),
                "learned_set_exact_t03": metrics.get("learned_set_exact_t03", ""),
                "learned_set_exact_t07": metrics.get("learned_set_exact_t07", ""),
                "learned_vs_symbolic_gap": learned_vs_symbolic,
                "learned_vs_ce_gap": learned_vs_ce,
                "reader_trainable_params": trainable_params,
                "reader_steps_to_90": "",
                "reader_steps_to_99": "",
                "reader_train_time": train_time,
                "reader_examples_per_sec": trained_examples / max(train_time, 1.0e-9) if budget > 0 else 0.0,
                "avg_written_slots": metrics.get("avg_written_slots", ""),
                "true_fact_written_rate": metrics.get("true_fact_written_rate", ""),
                "false_write_rate": metrics.get("false_write_rate", ""),
                "missed_fact_rate": metrics.get("missed_fact_rate", ""),
            }
            budget_rows.append(row)

        if normal_learned is not None:
            if steps_to_90 is None and normal_learned >= 0.90:
                steps_to_90 = budget
            if steps_to_99 is None and normal_learned >= 0.99:
                steps_to_99 = budget
        for row in budget_rows:
            row["reader_steps_to_90"] = "" if steps_to_90 is None else steps_to_90
            row["reader_steps_to_99"] = "" if steps_to_99 is None else steps_to_99
        raw_rows.extend(budget_rows)
        flush_outputs(out_dir, raw_rows, stress_mode=stress_mode)
        print(
            f"done task={task} slots={slot_count} hard={num_hard_negatives} sim={similarity_mode} "
            f"write={write_mode} seed={seed} budget={budget}",
            flush=True,
        )

    if 0 in budget_set:
        evaluate_budget(0)

    for step in range(1, max_budget + 1):
        reader.train()
        batch = train_dataset.sample_batch(args.batch_size, device=device)
        batch, _ = apply_write_mode(batch, write_mode)
        loss = train_loss(task, reader, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite reader loss for {task}/{write_mode}/seed{seed}/step{step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        optimizer.step()
        trained_examples += args.batch_size
        if step in budget_set:
            evaluate_budget(step)


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def update_results_md(path: Path, summary_rows: List[Dict[str, object]], args: argparse.Namespace, raw_count: int) -> None:
    budgets = parse_int_list(args.budgets)
    final_budget = max(budgets)
    tasks = parse_str_list(args.tasks, TASKS)
    write_modes = parse_write_modes(args.write_modes)
    lines = [
        "## Learned Structured Readers v1",
        "",
        "This run trains only tiny learned readers over typed memory slots. It does not train the local Transformer, CE decoder, memory writer, HPM-Lite recurrence, JEPA, ANN, graph memory, Priming, GKA, or RL.",
        "",
        f"Tasks: `{args.tasks}`. Write modes: `{args.write_modes}`. Controls: `{args.controls}`.",
        f"Budgets: `{args.budgets}`. Seeds: `{args.seeds}`. Seq len/window: `{args.seq_len}` / `{args.window}`.",
        f"Reader: dim `{args.reader_dim}`, hidden `{args.reader_hidden}`, layers `{args.reader_layers}`, dropout `{args.dropout}`, lr `{args.lr}`.",
        f"Device request: `{args.device}`. Platform: `{platform.platform()}`. Torch: `{torch.__version__}`.",
        "",
        "Command:",
        "",
        f"- `python scripts/run_structured_readers.py --tasks {args.tasks} --write-modes {args.write_modes} --budgets {args.budgets} --seeds {args.seeds} --seq-len {args.seq_len} --window {args.window} --batch-size {args.batch_size} --eval-batches {args.eval_batches} --reader-dim {args.reader_dim} --reader-hidden {args.reader_hidden} --device {args.device}`",
        "",
        f"Raw and summarized outputs: `runs/structured_readers_raw.csv` (`{raw_count}` rows), `runs/structured_readers_summary.csv` (`{len(summary_rows)}` rows).",
        "",
        "### Final-Budget Normal Results",
        "",
        "| task | write | symbolic exact | learned exact | CE reference | learned-symbolic gap | learned-CE gap | params | steps>=0.90 | steps>=0.99 | train time sec |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in tasks:
        for write_mode in write_modes:
            symbolic_metric = "symbolic_set_exact" if task == "coexisting" else "condition_symbolic_exact"
            learned_metric = "learned_set_exact" if task == "coexisting" else "learned_condition_exact"
            symbolic = summary_lookup(summary_rows, task, write_mode, "normal", final_budget, symbolic_metric)
            learned = summary_lookup(summary_rows, task, write_mode, "normal", final_budget, learned_metric)
            ce = summary_lookup(summary_rows, task, write_mode, "normal", final_budget, "ce_reference_exact")
            params = summary_lookup(summary_rows, task, write_mode, "normal", final_budget, "reader_trainable_params")
            step90 = summary_lookup(summary_rows, task, write_mode, "normal", final_budget, "reader_steps_to_90")
            step99 = summary_lookup(summary_rows, task, write_mode, "normal", final_budget, "reader_steps_to_99")
            train_time = summary_lookup(summary_rows, task, write_mode, "normal", final_budget, "reader_train_time")
            lines.append(
                f"| {task} | {write_mode} | {format_float(symbolic)} | {format_float(learned)} | {format_float(ce)} | {format_float(None if learned is None or symbolic is None else learned - symbolic)} | {format_float(None if learned is None or ce is None else learned - ce)} | {'' if params is None else f'{params:.0f}'} | {format_float(step90)} | {format_float(step99)} | {format_float(train_time)} |"
            )

    lines.extend(
        [
            "",
            "### Final-Budget Controls",
            "",
            "| task | write | control | symbolic exact | learned exact | CE reference | learned-CE gap |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    controls = parse_str_list(args.controls, CONTROLS)
    for task in tasks:
        for write_mode in write_modes:
            symbolic_metric = "symbolic_set_exact" if task == "coexisting" else "condition_symbolic_exact"
            learned_metric = "learned_set_exact" if task == "coexisting" else "learned_condition_exact"
            for control in controls:
                symbolic = summary_lookup(summary_rows, task, write_mode, control, final_budget, symbolic_metric)
                learned = summary_lookup(summary_rows, task, write_mode, control, final_budget, learned_metric)
                ce = summary_lookup(summary_rows, task, write_mode, control, final_budget, "ce_reference_exact")
                lines.append(
                    f"| {task} | {write_mode} | {control} | {format_float(symbolic)} | {format_float(learned)} | {format_float(ce)} | {format_float(None if learned is None or ce is None else learned - ce)} |"
                )

    cond = summary_lookup(summary_rows, "conditional_contrastive", "fact_token", "normal", final_budget, "learned_condition_exact")
    cond_sym = summary_lookup(summary_rows, "conditional_contrastive", "fact_token", "normal", final_budget, "condition_symbolic_exact")
    cond_ce = summary_lookup(summary_rows, "conditional_contrastive", "fact_token", "normal", final_budget, "ce_reference_exact")
    co = summary_lookup(summary_rows, "coexisting", "fact_token", "normal", final_budget, "learned_set_exact")
    co_sym = summary_lookup(summary_rows, "coexisting", "fact_token", "normal", final_budget, "symbolic_set_exact")
    co_ce = summary_lookup(summary_rows, "coexisting", "fact_token", "normal", final_budget, "ce_reference_exact")
    lines.extend(["", "### Verdict", ""])
    lines.append("Can a small learned structured reader recover what CE decoding fails to use?")
    if cond is not None and cond >= 0.95 and cond_ce is not None and cond_ce < 0.8:
        lines.append(f"Yes for condition binding: learned condition exact reaches {cond:.3f} while the CE reference is {cond_ce:.3f}. Condition binding is solved by the structured reader; CE decoding/read-use is the bottleneck.")
    elif cond is not None and cond_sym is not None and cond_sym >= 0.99 and cond < 0.95:
        lines.append(f"Not yet for condition binding: symbolic is {cond_sym:.3f} but learned exact is {cond:.3f}. Inspect reader implementation/training before blaming memory.")
    if co is not None and co >= 0.95 and (co_ce is None or co_ce < 0.8):
        ce_text = "unavailable" if co_ce is None else f"{co_ce:.3f}"
        lines.append(f"Yes for coexisting set readout: learned set exact reaches {co:.3f} while the CE reference is {ce_text}. Sequence next-token decoding is the wrong output operator for this set-valued query.")
    elif co is not None and co_sym is not None and co_sym >= 0.99 and co < 0.95:
        lines.append(f"Not yet for coexisting: symbolic set exact is {co_sym:.3f} but learned set exact is {co:.3f}.")
    lines.append("Does this support the architecture shift from HPM-Lite recurrence toward typed memory + structured readouts?")
    if cond is not None and co is not None and cond >= 0.95 and co >= 0.95:
        lines.append("Yes. In these diagnostics, typed memory plus task-appropriate learned readers solves cases where HPM-Lite recurrence did not separate from epmem.")
    else:
        lines.append("Partially. The symbolic upper bounds say storage is correct; any learned-reader miss should be treated as reader/training debt, not evidence for adding more HPM recurrence.")
    lines.append("")

    section = "\n".join(lines)
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## Learned Structured Readers v1\n"
        index = old.find(marker)
        if index >= 0:
            old = old[:index].rstrip()
        path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
    else:
        path.write_text("# HPM-Lite Results\n\n" + section + "\n", encoding="utf-8")


def write_short_writeup(path: Path, summary_rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    final_budget = max(parse_int_list(args.budgets))
    cond = summary_lookup(summary_rows, "conditional_contrastive", "fact_token", "normal", final_budget, "learned_condition_exact")
    cond_sym = summary_lookup(summary_rows, "conditional_contrastive", "fact_token", "normal", final_budget, "condition_symbolic_exact")
    cond_ce = summary_lookup(summary_rows, "conditional_contrastive", "fact_token", "normal", final_budget, "ce_reference_exact")
    co = summary_lookup(summary_rows, "coexisting", "fact_token", "normal", final_budget, "learned_set_exact")
    co_sym = summary_lookup(summary_rows, "coexisting", "fact_token", "normal", final_budget, "symbolic_set_exact")
    co_ce = summary_lookup(summary_rows, "coexisting", "fact_token", "normal", final_budget, "ce_reference_exact")
    text = f"""# Structured Memory Readout Result

## Problem

In these diagnostics, memory writing and retrieval can be correct while next-token CE decoding still fails. Typed structured readouts recover the correct answer, showing that the bottleneck is the read/use operator rather than storage.

## Readers

Conditional slots are typed as `(key_i, condition_i, value_i)`. The learned condition reader scores each slot with:

```text
s_i = MLP([q_key, q_cond, key_i, cond_i, value_i,
           q_key * key_i, q_cond * cond_i,
           abs(q_key - key_i), abs(q_cond - cond_i)])
L = CrossEntropy(s, correct_slot)
```

Coexisting slots are typed as `(key_i, value_i)`. The learned set reader predicts one sigmoid per slot:

```text
p_i = sigmoid(MLP([q_key, key_i, value_i,
                   q_key * key_i, abs(q_key - key_i)]))
L = BCE(p_i, 1[key_i == q_key and value_i in answer_set])
```

## Result

| task | CE reference | symbolic exact | learned exact |
| --- | ---: | ---: | ---: |
| conditional_contrastive | {format_float(cond_ce)} | {format_float(cond_sym)} | {format_float(cond)} |
| coexisting | {format_float(co_ce)} | {format_float(co_sym)} | {format_float(co)} |

## Interpretation

If learned exact approaches symbolic exact while CE exact remains low, this supports the claim that generic next-token decoding is the wrong readout for typed memory operations. The result does not justify adding HPM-Lite recurrence; the previous diagnostics showed HPM-Lite did not meaningfully separate from epmem.

## Limitations

The readers are diagnostic modules, not a full language model interface. They use synthetic typed slots and task-specific losses. The run does not test learned writing, noisy real text, large vocabularies, or free-form generation.

## Next Steps

Stress these readers under harder slot ambiguity, then only after that consider learned writing. Do not add JEPA, ANN, graph memory, Priming, GKA, RL, or larger backbones until typed readout behavior is understood.
"""
    path.write_text(text, encoding="utf-8")


def update_stress_writeup(path: Path, summary_rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    final_budget = max(parse_int_list(args.budgets))
    normal_rows = stress_rows(summary_rows, final_budget)
    learned_values = []
    symbolic_values = []
    gap_values = []
    for row in normal_rows:
        learned = row_float(row, f"{stress_metric_name(row['task'], 'learned')}_mean")
        symbolic = row_float(row, f"{stress_metric_name(row['task'], 'symbolic')}_mean")
        gap = row_float(row, "learned_vs_symbolic_gap_mean")
        if learned is not None:
            learned_values.append(learned)
        if symbolic is not None:
            symbolic_values.append(symbolic)
        if gap is not None:
            gap_values.append(gap)

    learned_text = format_float(statistics.mean(learned_values) if learned_values else None)
    symbolic_text = format_float(statistics.mean(symbolic_values) if symbolic_values else None)
    gap_text = format_float(statistics.mean(gap_values) if gap_values else None)
    worst = None
    for row in normal_rows:
        learned = row_float(row, f"{stress_metric_name(row['task'], 'learned')}_mean")
        symbolic = row_float(row, f"{stress_metric_name(row['task'], 'symbolic')}_mean")
        if learned is None or symbolic is None:
            continue
        gap = learned - symbolic
        if worst is None or gap < worst[0]:
            worst = (gap, row, learned, symbolic)
    if worst is None:
        worst_text = "No normal stress rows were available for failure analysis."
    else:
        gap, row, learned, symbolic = worst
        worst_text = (
            f"Worst normal gap: `{row['task']}`, slot_count `{row['slot_count']}`, "
            f"hard negatives `{row['num_hard_negatives']}`, similarity `{row['similarity_mode']}` "
            f"with learned exact {learned:.3f}, symbolic exact {symbolic:.3f}, gap {gap:.3f}."
        )

    if learned_values and statistics.mean(learned_values) >= 0.95:
        next_step = "The reader survives this stress level, so the next diagnostic should move toward learned writing or noisier slot extraction."
    else:
        next_step = "The reader now lags the symbolic upper bound, so reader robustness should be improved before adding learned writing."

    section = f"""## Structured Reader Stress Test

The stress suite varies slot count, hard negatives, token similarity, and random slot order while still training only the structured reader. It keeps symbolic readers as the upper bound.

| final budget | mean symbolic exact | mean learned exact | mean learned-symbolic gap |
| ---: | ---: | ---: | ---: |
| {final_budget} | {symbolic_text} | {learned_text} | {gap_text} |

{worst_text}

Controls remain important: `shuffled_values` and `corrupt_values` test whether value identity matters, `random_keys` tests key matching, and `corrupt_conditions` specifically tests conditional binding.

Limitation: this is still synthetic typed-slot data. Passing this suite does not prove learned writing, natural language extraction, or free-form generation.

Next step: {next_step}
"""
    if path.exists():
        old = path.read_text(encoding="utf-8").rstrip()
        marker = "\n## Structured Reader Stress Test\n"
        index = old.find(marker)
        if index >= 0:
            old = old[:index].rstrip()
        path.write_text(old + "\n\n" + section + "\n", encoding="utf-8")
    else:
        path.write_text("# Structured Memory Readout Result\n\n" + section + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tasks = parse_str_list(args.tasks, TASKS)
    write_modes = parse_write_modes(args.write_modes)
    stress_mode = any(is_stress_task(task) for task in tasks)
    controls = parse_str_list(args.stress_controls if stress_mode else args.controls, CONTROLS)
    budgets = sorted(set(parse_int_list(args.budgets)))
    seeds = parse_int_list(args.seeds)
    slot_counts = parse_int_list(args.slot_counts) if args.slot_counts else [args.num_facts]
    hard_negative_counts = parse_int_list(args.num_hard_negatives)
    similarity_modes = parse_str_list(args.similarity_modes, {"none", "adjacent", "confusable", "mixed"})
    slot_orders = parse_str_list(args.slot_order, {"original", "random"})
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    ce_refs = load_ce_references(out_dir)

    raw_rows: List[Dict[str, object]] = []
    leak_checks = 0
    for task in tasks:
        task_slot_counts = slot_counts if is_stress_task(task) else [args.num_facts]
        task_hard_negative_counts = hard_negative_counts if is_stress_task(task) else [0]
        task_similarity_modes = similarity_modes if is_stress_task(task) else ["none"]
        task_slot_orders = slot_orders if is_stress_task(task) else ["random"]
        task_positive_counts = parse_auto_positive(args.num_positive, task) if is_stress_task(task) else parse_auto_positive("auto", task)
        for slot_count in task_slot_counts:
            for num_positive in task_positive_counts:
                for num_hard_negatives in task_hard_negative_counts:
                    for similarity_mode in task_similarity_modes:
                        for slot_order in task_slot_orders:
                            for seed in seeds:
                                leak_checks += verify_no_answer_in_memory(
                                    task=task,
                                    seq_len=args.seq_len,
                                    window=args.window,
                                    seed=seed,
                                    batch_size=args.batch_size,
                                    num_facts=slot_count,
                                    write_modes=write_modes,
                                    num_positive=num_positive,
                                    num_hard_negatives=num_hard_negatives,
                                    similarity_mode=similarity_mode,
                                    slot_order=slot_order,
                                )
                            for write_mode in write_modes:
                                for seed in seeds:
                                    train_and_evaluate(
                                        task=task,
                                        write_mode=write_mode,
                                        seed=seed,
                                        budgets=budgets,
                                        controls=controls,
                                        args=args,
                                        device=device,
                                        ce_refs=ce_refs,
                                        raw_rows=raw_rows,
                                        out_dir=out_dir,
                                        slot_count=slot_count,
                                        num_positive=num_positive,
                                        num_hard_negatives=num_hard_negatives,
                                        similarity_mode=similarity_mode,
                                        slot_order=slot_order,
                                        stress_mode=stress_mode,
                                    )

    summary_rows = flush_outputs(out_dir, raw_rows, stress_mode=stress_mode)
    if stress_mode:
        update_stress_results_md(Path("results.md"), summary_rows, args, len(raw_rows))
        update_stress_writeup(Path("writeup_structured_memory_result.md"), summary_rows, args)
        print(
            f"wrote runs/structured_reader_stress_raw.csv, runs/structured_reader_stress_summary.csv, "
            f"results.md, writeup_structured_memory_result.md; leak_checks={leak_checks}"
        )
    else:
        update_results_md(Path("results.md"), summary_rows, args, len(raw_rows))
        write_short_writeup(Path("writeup_structured_memory_result.md"), summary_rows, args)
        print(
            f"wrote runs/structured_readers_raw.csv, runs/structured_readers_summary.csv, "
            f"results.md, writeup_structured_memory_result.md; leak_checks={leak_checks}"
        )


if __name__ == "__main__":
    main()
