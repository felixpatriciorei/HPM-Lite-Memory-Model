# HPM-Lite v2 retrieval schema fix

Tiny learned-writer smoke tests can temporarily select no valid target fact. In that case
`retrieval_metrics(...)` returns an empty dictionary because there is no valid positive
slot to score against. That made early HPM-Lite v2 runs drop `eval_retrieval_top1` from
the metrics dictionary.

This patch treats that situation as an explicit retrieval failure (`0.0`) whenever the
model produced retrieval `top_indices`. Local models still do not get retrieval metrics.
