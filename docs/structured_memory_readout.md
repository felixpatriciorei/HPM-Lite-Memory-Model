# Structured Memory Readout Result

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
| conditional_contrastive | 0.5600 | 1.0000 | 1.0000 |
| coexisting | 0.0100 | 1.0000 | 1.0000 |

The CE references are taken from earlier CE diagnostics; the structured-reader run trains only the reader modules.

## Interpretation

If learned exact approaches symbolic exact while CE exact remains low, this supports the claim that generic next-token decoding is the wrong readout for typed memory operations. The result does not justify adding HPM-Lite recurrence; the previous diagnostics showed HPM-Lite did not meaningfully separate from epmem.

## Limitations

The readers are diagnostic modules, not a full language model interface. They use synthetic typed slots and task-specific losses. The run does not test learned writing, noisy real text, large vocabularies, or free-form generation.

## Next Steps

Stress these readers under harder slot ambiguity, then only after that consider learned writing. Do not add JEPA, ANN, graph memory, Priming, GKA, RL, or larger backbones until typed readout behavior is understood.

## Structured Reader Stress Test

The stress suite varies slot count, hard negatives, token similarity, and random slot order while still training only the structured reader. It keeps symbolic readers as the upper bound.

| final budget | mean symbolic exact | mean learned exact | mean learned-symbolic gap |
| ---: | ---: | ---: | ---: |
| 100 | 1.0000 | 1.0000 | 0.0000 |

Worst normal gap: `coexisting_stress`, slot_count `4`, hard negatives `0`, similarity `mixed` with learned exact 1.000, symbolic exact 1.000, gap 0.000.

This is the Stage A smoke grid only. It covered slot counts 4 and 16, hard negatives 0 and 8, similarity modes none and mixed, seeds 0 and 1, and budgets 0, 10, and 100. The full Stage B grid was not run because Stage A took about 11 minutes and Stage B was projected to take several hours.

Controls remain important: `shuffled_values` and `corrupt_values` test whether value identity matters, `random_keys` tests key matching, and `corrupt_conditions` specifically tests conditional binding.

Limitation: this is still synthetic typed-slot data. Passing this suite does not prove learned writing, natural language extraction, or free-form generation.

Next step: The reader survives this stress level, so the next diagnostic should move toward learned writing or noisier slot extraction.

## Noisy Slot Extraction / Learned Writer v1

The next diagnostic replaces exact `FACT` parsing with noisy synthetic templates and a small learned typed extractor. The extractor is a pointer model over pre-query tokens; it predicts key/value/condition positions for each known slot in canonical occurrence order.

| final budget | learned extractor slot F1 | fact_token exact at marker 0.0 | learned extractor exact at marker 0.0 |
| ---: | ---: | ---: | ---: |
| 100 | 0.7793 | 0.0000 | 0.6391 |

Interpretation: oracle slots test the reader upper bound, fact-token slots test brittle marker parsing, and learned slots test whether typed memory can move beyond hand-coded `FACT` extraction. V1 still assumes the number of slots is known and uses canonical slot order instead of Hungarian matching.

## Noisy Slot Extraction v2: Order-Invariant Writer

V1 was useful but too friendly: it assumed the true slot count and trained against canonical occurrence order. V2 replaces that with a small DETR-style set extractor. It emits up to `max_slots` unordered slot queries, predicts objectness for each query, and points to typed key/value/condition positions inside the pre-query sequence.

| final budget | fact_token exact at marker 0.0 | v1 canonical exact at marker 0.0 | v2 unordered exact at marker 0.0 | v2 slot F1 | v2 all-slots exact | v2 slot-count accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.0000 | 0.6617 | 0.7152 | 0.8319 | 0.5145 | 0.6734 |

The important distinction is that v2 no longer chooses slot `j` because the metadata says the `j`th gold fact exists. It must decide whether each predicted slot exists, then form a typed tuple from pointers. This is a more honest bridge from hand-coded parsing to learned writing.

Remaining limitations: Stage A still trains separate configurations for fixed slot-count settings, so variable-count generalization is only partially tested. The assignment matcher is an exact local Hungarian-style solver for the rectangular matching used here, but the data is still synthetic token text rather than real extraction.

Next recommendation: if v2 slot-count accuracy is low, improve objectness/duplicate suppression before touching the reader. If slot F1 is low while oracle-slot reader exact is high, writer field extraction is the bottleneck. If v2 is strong at marker-rate 0.0, the next step is harder noisy writing or real-data slot extraction.

## Writer v2 Failure Decomposition

The v2 writer now has a bottleneck decomposition rather than a new architecture. The diagnostic compares normal objectness-threshold inference with oracle count, oracle objectness, oracle fields, oracle count+fields, and a threshold sweep.

| final budget | normal exact | oracle count exact | oracle objectness exact | oracle fields exact | objectness margin | slot-count accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.3981 | 0.3634 | 0.4966 | 0.7238 | 0.5211 | 0.4169 |

Interpretation rule: if oracle count or oracle objectness jumps, Writer v3 should target adaptive slot selection/calibration. If oracle fields jumps, Writer v3 should target better pointer fields or high-recall span candidates. If neither jumps, inspect reader integration or data generation before changing architecture.

## Writer v3 Field Candidate Decomposition

Writer v3 tests whether high-recall field candidates can repair the v2 pointer bottleneck. It decomposes the pipeline into candidate proposal and slot assembly over candidate fields.

| final budget | v2 exact | v3 oracle-candidate exact | v3 learned-candidate exact | learned candidate recall | v3 gain over v2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.4207 | 0.0254 | 0.0250 | 0.9286 | -0.3957 |

Interpretation rule: if oracle candidates work but learned candidates do not, field proposal recall is the bottleneck. If oracle candidates do not work, tuple assembly is the bottleneck. If learned candidates beat v2, high-recall candidate fields are moving the writer in the right direction.

## Tuple Assembly Failure Analysis

This diagnostic isolates the v3 tuple assembler under oracle candidate fields. It checks independent field heads, true-count/no-objectness assembly, gold-field ablations, and an SPN-style whole-tuple scorer.

| final budget | independent heads exact | true-count exact | gold-all-fields exact | SPN tuple exact |
| ---: | ---: | ---: | ---: | ---: |
| 300 | 0.0563 | 0.0563 | 1.0000 | 0.2937 |

Interpretation rule: if gold-all-fields is not 1.0, indexing or slot formatting is broken. If gold-all-fields is 1.0 but true-count independent heads stay low, the factorized assembler objective is failing. If SPN succeeds where independent heads fail, Writer v4 should move toward whole-tuple scoring.

## Contextual Tuple Assembly

Writer v4 tests whether tuple assembly needs relation evidence from the original text, not just a product of candidate field embeddings. The scorer sees contextual token states, relative field positions, field order, pooled text between fields, and local windows around key/condition/value candidates.

| final budget | contextual oracle exact | contextual hard-negative exact | gold key+cond exact | gold-all-fields exact | hard-negative score margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 19.6931 |

Interpretation rule: if oracle candidates reach 1.0, contextual relation evidence fixed tuple assembly under clean candidates. If oracle candidates work but hard negatives fail, relation discrimination is the bottleneck. If gold key+condition is still low, value relation scoring is weak or labels/features are still wrong.

## Learned Candidate Extraction Bottleneck

After contextual tuple assembly solved oracle candidates, this diagnostic swaps in candidate fields from the learned `CandidateFieldProposer`. Repair/noise/oracle-field modes separate missing candidates from noisy candidate pools and tuple-scoring failures.

| final budget | oracle exact | learned exact | learned+oracle-missing exact | oracle+learned-noise exact | key recall | condition recall | value recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 |  | 0.3125 |  |  | 1.0000 | 1.0000 | 1.0000 |

Interpretation rule: if learned+oracle-missing jumps, candidate recall is the bottleneck. If oracle+learned-noise falls, candidate false positives are breaking tuple scoring. If recall is high but learned exact is low, tuple selection under noisy learned pools is the bottleneck.

## Integrated Memory Model v1

The integrated comparison moves from isolated writer/reader diagnostics to a model-level baseline: a tiny local Transformer trained with answer CE versus Writer v4.5 plus typed slots and a learned structured reader.

| final budget | mean memory gain vs Transformer | worst integrated exact |
| ---: | ---: | ---: |
| 1000 |  |  |

Interpretation: a clear positive gain means typed memory is useful as a model component on these synthetic noisy tasks, but a weak worst-case cell means the next step is integration/debugging rather than model scaling.

