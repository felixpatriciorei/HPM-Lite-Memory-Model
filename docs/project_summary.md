# Project Summary

HPM-Lite is a small PyTorch research repo for controlled memory diagnostics. The main idea is to separate:

1. whether a model can store/retrieve long-range facts,
2. whether the readout operator can use retrieved memory correctly,
3. whether apparent success survives controls such as shuffled values, random keys, no retrieval, and random writes.

The current code is intended for research and portfolio review, not production deployment.
