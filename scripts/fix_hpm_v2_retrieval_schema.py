from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already patched: {path}")
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected block in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched: {path}")


def patch_evaluate() -> None:
    path = ROOT / "hpm_lite" / "evaluate.py"
    patch_once(
        path,
        "    ret_margin_sum = 0.0\n    ret_count = 0\n",
        "    ret_margin_sum = 0.0\n    ret_count = 0\n    saw_retrieval_output = False\n",
    )
    patch_once(
        path,
        "        output = model(\n"
        "            batch[\"input_ids\"],\n"
        "            memory_token_positions=batch[\"memory_token_positions\"],\n"
        "            memory_mask=batch[\"memory_mask\"],\n"
        "            answer_positions=batch[\"answer_positions\"],\n"
        "            query_key_positions=batch[\"query_key_positions\"],\n"
        "            top_k=top_k,\n"
        "            task=task,\n"
        "            hop_positive_memory_indices=batch[\"hop_positive_memory_indices\"],\n"
        "            memory_control=memory_control,\n"
        "            use_learned_writer=use_learned_writer,\n"
        "            learned_writer_teacher_forcing=False,\n"
        "        )\n",
        "        output = model(\n"
        "            batch[\"input_ids\"],\n"
        "            memory_token_positions=batch[\"memory_token_positions\"],\n"
        "            memory_mask=batch[\"memory_mask\"],\n"
        "            answer_positions=batch[\"answer_positions\"],\n"
        "            query_key_positions=batch[\"query_key_positions\"],\n"
        "            top_k=top_k,\n"
        "            task=task,\n"
        "            hop_positive_memory_indices=batch[\"hop_positive_memory_indices\"],\n"
        "            memory_control=memory_control,\n"
        "            use_learned_writer=use_learned_writer,\n"
        "            learned_writer_teacher_forcing=False,\n"
        "        )\n"
        "        if output.get(\"retrieval\") and \"top_indices\" in output[\"retrieval\"]:\n"
        "            saw_retrieval_output = True\n",
    )
    patch_once(
        path,
        "    if ret_count:\n"
        "        metrics.update(\n"
        "            {\n"
        "                \"retrieval_top1\": ret_top1_sum / ret_count,\n"
        "                \"retrieval_topk\": ret_topk_sum / ret_count,\n"
        "                \"retrieval_margin\": ret_margin_sum / ret_count,\n"
        "            }\n"
        "        )\n",
        "    if ret_count:\n"
        "        metrics.update(\n"
        "            {\n"
        "                \"retrieval_top1\": ret_top1_sum / ret_count,\n"
        "                \"retrieval_topk\": ret_topk_sum / ret_count,\n"
        "                \"retrieval_margin\": ret_margin_sum / ret_count,\n"
        "            }\n"
        "        )\n"
        "    elif saw_retrieval_output:\n"
        "        # Learned writers can temporarily write no valid target facts during early\n"
        "        # training. The retrieval module still ran, so report retrieval as 0\n"
        "        # instead of dropping the schema key entirely.\n"
        "        metrics.update({\"retrieval_top1\": 0.0, \"retrieval_topk\": 0.0, \"retrieval_margin\": 0.0})\n",
    )


def patch_train() -> None:
    path = ROOT / "hpm_lite" / "train.py"
    patch_once(
        path,
        "                ret = retrieval_metrics(\n"
        "                    output[\"retrieval\"],\n"
        "                    positive_indices=metric_batch[\"positive_memory_indices\"],\n"
        "                    positive_mask=metric_batch.get(\"positive_memory_mask\"),\n"
        "                )\n",
        "                ret = retrieval_metrics(\n"
        "                    output[\"retrieval\"],\n"
        "                    positive_indices=metric_batch[\"positive_memory_indices\"],\n"
        "                    positive_mask=metric_batch.get(\"positive_memory_mask\"),\n"
        "                )\n"
        "                if not ret and output.get(\"retrieval\") and \"top_indices\" in output[\"retrieval\"]:\n"
        "                    # Early learned-writer runs may retrieve from a writer-selected\n"
        "                    # memory set that contains no target fact. Treat that as a\n"
        "                    # measurable retrieval failure, not as a missing schema field.\n"
        "                    ret = {\"retrieval_top1\": 0.0, \"retrieval_topk\": 0.0, \"retrieval_margin\": 0.0}\n",
    )


def main() -> None:
    patch_evaluate()
    patch_train()
    print("HPM-Lite v2 retrieval schema fallback enabled")


if __name__ == "__main__":
    main()
