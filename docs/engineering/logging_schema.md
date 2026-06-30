# HPM-Lite Logging Schema

Research-grade figures require better logs.

## `results/raw/run_summary.csv`

One row per completed run.

Columns:

```csv
run_id,commit_hash,timestamp,device,gpu_name,torch_version,cuda_available,model,model_variant,task,write_mode,seq_len,window,d_model,layers,heads,batch_size,steps,seed,parameters,train_wall_time_sec,eval_examples,examples_per_sec,tokens_per_sec,peak_vram_mb,eval_answer_exact,eval_answer_ce,eval_retrieval_top1,eval_retrieval_topk,eval_true_fact_written_rate,eval_false_write_rate,eval_missed_fact_rate,eval_avg_written_slots,eval_retrieval_margin
```

## `results/raw/step_log.csv`

One row per logging interval.

```csv
run_id,step,model,model_variant,seq_len,seed,train_loss,train_answer_exact,train_answer_ce,train_retrieval_top1,train_writer_recall,train_false_write_rate,train_missed_fact_rate,eval_answer_exact,eval_answer_ce,eval_retrieval_top1,eval_writer_recall,examples_per_sec_recent,tokens_per_sec_recent,peak_vram_mb
```

## `results/traces/retrieval_trace.csv`

One row per evaluated query.

```csv
run_id,example_id,seq_len,seed,query_key,gold_value,answer_correct,correct_slot_id,top1_slot_id,correct_slot_rank,correct_slot_score,best_wrong_slot_score,null_slot_score,retrieval_margin,topk_slot_ids,topk_scores
```

## `results/traces/router_trace.csv`

One row per sampled token position.

```csv
run_id,example_id,position,token,token_type,alpha_local,alpha_recurrent,alpha_episodic,router_entropy,answer_correct
```

## `results/traces/memory_graph_nodes.csv`

```csv
run_id,example_id,node_id,node_type,label,position,is_gold
```

## `results/traces/memory_graph_edges.csv`

```csv
run_id,example_id,src_node,dst_node,edge_type,score,is_correct
```

## `results/processed/ablation_summary.csv`

```csv
model_variant,seq_len,seed,exact,ce,retrieval_top1,writer_recall,false_write_rate,missed_fact_rate,params,peak_vram_mb,tokens_per_sec
```

## Rule

Do not plot confidence intervals from one seed.
If there is only one seed, label the figure as a pilot result.
