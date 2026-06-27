from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import torch

from .data import FactRecallConfig, FactRecallDataset, VOCAB_SIZE
from .evaluate import evaluate_batches
from .metrics import answer_cross_entropy, answer_span_exact_accuracy, count_parameters, retrieval_metrics
from .model import HpmLiteConfig, HpmLiteModel
from .utils import ensure_dir, resolve_device, set_seed, str_to_bool, timestamp, write_json
from .write_modes import apply_write_mode, batch_from_memory_selection, writer_metrics


class TinyAdamW:
    """Small AdamW optimizer to avoid torch._dynamo imports in broken CPU installs."""

    def __init__(
        self,
        parameters,
        lr: float = 3.0e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.01,
    ):
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.parameters]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            grad = parameter.grad
            if self.weight_decay:
                parameter.mul_(1.0 - self.lr * self.weight_decay)
            self.m[index].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            self.v[index].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            denom = self.v[index].sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
            step_size = self.lr / bias_correction1
            parameter.addcdiv_(self.m[index], denom, value=-step_size)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal HPM-Lite synthetic recall model.")
    parser.add_argument("--model", choices=["local", "recurrent", "epmem", "hpm_lite", "hebbian"], default="local")
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
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lambda-ret", type=float, default=0.1)
    parser.add_argument("--lambda-writer", type=float, default=0.1)
    parser.add_argument("--learned-writer-teacher-forcing-steps", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--memory-null-slot", type=str_to_bool, default=False)
    parser.add_argument("--null-score-init", type=float, default=0.0)
    parser.add_argument(
        "--memory-control",
        choices=["normal", "shuffle_values", "shuffled_values", "random_keys", "corrupt_values", "no_retrieval"],
        default="normal",
    )
    parser.add_argument("--write-mode", choices=["oracle", "fact_token", "random_write", "learned"], default="oracle")
    parser.add_argument("--oracle-memory", type=str_to_bool, default=True)
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument("--repeated-keys", type=str_to_bool, default=False)
    parser.add_argument("--similar-values", type=str_to_bool, default=False)
    parser.add_argument("--distractor-fact-spans", type=int, default=0)
    parser.add_argument("--query-key-noise-only", type=str_to_bool, default=False)
    parser.add_argument("--fact-order", choices=["random", "query_last"], default="random")
    parser.add_argument("--out-dir", type=str, default="runs")
    parser.add_argument("--save-checkpoint", type=str_to_bool, default=True)
    return parser


def args_with_defaults(args: argparse.Namespace | SimpleNamespace) -> argparse.Namespace:
    defaults = vars(build_arg_parser().parse_args([]))
    merged = {**defaults, **vars(args)}
    return argparse.Namespace(**merged)


def make_model(args: argparse.Namespace, device: torch.device) -> HpmLiteModel:
    config = HpmLiteConfig(
        model_type=args.model,
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        window=args.window,
        max_seq_len=max(2048, args.seq_len + 1),
        use_null_slot=args.memory_null_slot,
        null_score_init=args.null_score_init,
        use_learned_writer=args.write_mode == "learned",
    )
    return HpmLiteModel(config).to(device)


def forward_batch(model: HpmLiteModel, batch: Dict[str, torch.Tensor], args: argparse.Namespace) -> Dict[str, Any]:
    batch, _ = apply_write_mode(batch, args.write_mode)
    return model(
        batch["input_ids"],
        memory_token_positions=batch["memory_token_positions"],
        memory_mask=batch["memory_mask"],
        answer_positions=batch["answer_positions"],
        query_key_positions=batch["query_key_positions"],
        top_k=args.top_k,
        task=args.task,
        hop_positive_memory_indices=batch["hop_positive_memory_indices"],
        memory_control=args.memory_control,
        use_learned_writer=args.write_mode == "learned",
        learned_writer_teacher_forcing=False,
    )


def run_training(args: argparse.Namespace | SimpleNamespace) -> Dict[str, Any]:
    args = args_with_defaults(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=args.seq_len,
            window=args.window,
            task=args.task,
            num_facts=args.num_facts,
            seed=args.seed,
            oracle_memory=args.oracle_memory,
            repeated_keys=args.repeated_keys,
            similar_values=args.similar_values,
            distractor_fact_spans=args.distractor_fact_spans,
            query_key_noise_only=args.query_key_noise_only,
            fact_order=args.fact_order,
        )
    )
    eval_dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=args.seq_len,
            window=args.window,
            task=args.task,
            num_facts=args.num_facts,
            seed=args.seed + 100_000,
            oracle_memory=args.oracle_memory,
            repeated_keys=args.repeated_keys,
            similar_values=args.similar_values,
            distractor_fact_spans=args.distractor_fact_spans,
            query_key_noise_only=args.query_key_noise_only,
            fact_order=args.fact_order,
        )
    )
    model = make_model(args, device)
    optimizer = TinyAdamW(model.parameters(), lr=args.lr)

    run_dir = ensure_dir(Path(args.out_dir) / f"{timestamp()}_{args.model}_{args.task}_seed{args.seed}")
    start_time = time.perf_counter()
    last_time = start_time
    final_metrics: Dict[str, Any] = {}

    for step in range(1, args.steps + 1):
        model.train()
        batch = train_dataset.sample_batch(args.batch_size, device=device)
        batch, write_stats = apply_write_mode(batch, args.write_mode)
        learned_writer = args.write_mode == "learned"
        teacher_forcing = learned_writer and step <= args.learned_writer_teacher_forcing_steps
        output = model(
            batch["input_ids"],
            memory_token_positions=batch["memory_token_positions"],
            memory_mask=batch["memory_mask"],
            answer_positions=batch["answer_positions"],
            query_key_positions=batch["query_key_positions"],
            top_k=args.top_k,
            task=args.task,
            hop_positive_memory_indices=batch["hop_positive_memory_indices"],
            memory_control=args.memory_control,
            use_learned_writer=learned_writer,
            learned_writer_teacher_forcing=teacher_forcing,
        )
        metric_batch = batch
        if learned_writer and "writer_memory_token_positions" in output["retrieval"]:
            learned_write_batch = batch_from_memory_selection(
                batch,
                output["retrieval"]["writer_memory_token_positions"],
                output["retrieval"]["writer_memory_mask"],
            )
            write_stats = writer_metrics(batch, learned_write_batch)
            if not teacher_forcing:
                metric_batch = learned_write_batch
        logits = output["logits"]
        answer_loss = answer_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        retrieval_loss = output["retrieval"].get("retrieval_loss", logits.new_zeros(()))
        writer_loss = output["retrieval"].get("writer_loss", logits.new_zeros(()))
        loss = answer_loss + args.lambda_ret * retrieval_loss + args.lambda_writer * writer_loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        should_eval = step % args.eval_every == 0 or step == args.steps
        if should_eval:
            now = time.perf_counter()
            elapsed = now - last_time
            examples_per_sec = (args.batch_size * args.eval_every) / max(elapsed, 1.0e-9)
            last_time = now
            train_acc = answer_span_exact_accuracy(logits.detach(), batch["target_ids"], batch["loss_mask"])
            ret = retrieval_metrics(
                output["retrieval"],
                positive_indices=metric_batch["positive_memory_indices"],
                positive_mask=metric_batch.get("positive_memory_mask"),
            )
            eval_metrics = evaluate_batches(
                model=model,
                dataset=eval_dataset,
                batch_size=args.batch_size,
                batches=args.eval_batches,
                device=device,
                task=args.task,
                top_k=args.top_k,
                memory_control=args.memory_control,
                write_mode=args.write_mode,
                use_learned_writer=learned_writer,
            )
            final_metrics = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_answer_ce": float(answer_loss.item()),
                "train_answer_exact": float(train_acc.item()),
                "train_retrieval_loss": float(retrieval_loss.item()),
                "train_writer_loss": float(writer_loss.item()),
                "examples_per_sec_recent": examples_per_sec,
                **{f"train_writer_{key}": value for key, value in write_stats.items()},
                **{f"train_{key}": value for key, value in ret.items()},
                **{f"eval_{key}": value for key, value in eval_metrics.items()},
            }
            compact = {
                "step": step,
                "loss": round(final_metrics["train_loss"], 4),
                "eval_exact": round(final_metrics["eval_answer_exact"], 4),
                "eval_ce": round(final_metrics["eval_answer_ce"], 4),
            }
            if "eval_retrieval_top1" in final_metrics:
                compact["eval_ret_top1"] = round(final_metrics["eval_retrieval_top1"], 4)
            if learned_writer:
                compact["writer_recall"] = round(final_metrics.get("train_writer_true_fact_written_rate", 0.0), 4)
            print(json.dumps(compact, sort_keys=True))

    total_time = time.perf_counter() - start_time
    final_metrics.update(
        {
            "model": args.model,
            "task": args.task,
            "write_mode": args.write_mode,
            "seq_len": args.seq_len,
            "window": args.window,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "device": str(device),
            "parameters": count_parameters(model),
            "memory_slots_per_sample": 0 if args.model == "local" else train_dataset.config.num_facts,
            "train_wall_time_sec": total_time,
            "lambda_writer": args.lambda_writer,
            "learned_writer_teacher_forcing_steps": args.learned_writer_teacher_forcing_steps,
            "run_dir": str(run_dir),
        }
    )
    write_json(run_dir / "metrics.json", final_metrics)

    if args.save_checkpoint:
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": asdict(model.config),
                "args": vars(args),
                "metrics": final_metrics,
            },
            run_dir / "checkpoint.pt",
        )
    write_run_summary(run_dir / "summary.md", final_metrics, args)
    return final_metrics


def write_run_summary(path: Path, metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# HPM-Lite Run Summary",
        "",
        f"Command model: `{args.model}`",
        f"Task: `{args.task}`",
        f"Sequence length/window: `{args.seq_len}` / `{args.window}`",
        f"Final exact accuracy: `{metrics.get('eval_answer_exact', 0.0):.4f}`",
        f"Final answer CE: `{metrics.get('eval_answer_ce', 0.0):.4f}`",
        f"Wall-clock train time: `{metrics.get('train_wall_time_sec', 0.0):.2f}s`",
        "",
        "This single run is a sanity check, not a claim about the architecture.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    main()
