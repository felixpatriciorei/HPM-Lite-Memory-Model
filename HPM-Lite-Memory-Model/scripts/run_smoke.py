from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpm_lite.train import run_training


def main() -> None:
    results = []
    for index, model in enumerate(["local", "epmem", "hpm_lite", "hebbian"]):
        args = SimpleNamespace(
            model=model,
            task="kv",
            seq_len=192,
            window=64,
            batch_size=8,
            steps=5,
            eval_every=5,
            eval_batches=2,
            d_model=64,
            layers=1,
            heads=4,
            lr=3.0e-4,
            seed=1234 + index,
            device="cpu",
            lambda_ret=0.1,
            top_k=1,
            memory_control="normal",
            oracle_memory=True,
            out_dir="runs/smoke",
            save_checkpoint=False,
        )
        metrics = run_training(args)
        results.append(
            {
                "model": model,
                "eval_answer_exact": metrics.get("eval_answer_exact"),
                "eval_answer_ce": metrics.get("eval_answer_ce"),
                "eval_retrieval_top1": metrics.get("eval_retrieval_top1"),
            }
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
