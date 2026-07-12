# Research context for HPM-Lite v2 long-context matrix

HPM-Lite v2 is not trying to replace the Transformer literature. It is a small controlled memory testbed that asks whether explicit write/retrieve memory can solve exact long-range recall when local attention cannot see the original fact.

This experiment sits near several long-context architecture ideas:

- Transformer-XL addresses fixed-length context by adding segment-level recurrence and relative positional encoding.
- RetNet studies parallel, recurrent, and chunkwise recurrent computation modes for long-sequence modeling.
- Mamba uses input-selective state-space updates so the model can choose what information to propagate or forget over long sequences.
- Hyena studies subquadratic long convolution plus gating as an alternative to full attention at long sequence length.

The HPM-Lite v2 matrix is narrower than those systems. It focuses on one falsifiable failure question:

> If retrieval remains accurate but answer accuracy drops, is the failure caused by retrieval collapse or by the model failing to write the correct facts into memory?

The overnight matrix strongly points to the second answer in this benchmark. Retrieval top-1 remains near 1.0 across 4096, 8192, and 12288 tokens, while exact answer accuracy follows the true-fact written rate and the writer-supervision schedule.

References:

- Dai et al., 2019, Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, https://arxiv.org/abs/1901.02860
- Sun et al., 2023, Retentive Network: A Successor to Transformer for Large Language Models, https://arxiv.org/abs/2307.08621
- Gu and Dao, 2023, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, https://arxiv.org/abs/2312.00752
- Poli et al., 2023, Hyena Hierarchy: Towards Larger Convolutional Language Models, https://arxiv.org/abs/2302.10866
