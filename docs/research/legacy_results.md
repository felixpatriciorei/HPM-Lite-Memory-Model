# HPM-Lite Results

## Stage 2 Write-Mode Validation

Models: `local`, `epmem`, `hpm_lite`, `hebbian`.
Task: `kv`. Write modes: `oracle,fact_token,random_write`.
Seeds: `0,1,2,3,4`. Sequence lengths: `256,512,1024`. Window: `64`.
Steps: `5`. Batch size: `4`. Eval batches: `2`.
Device request: `cpu`. Platform: `Windows-10-10.0.19045-SP0`. Torch: `2.12.0+cpu`.
Leak checks passed on `180` generated examples; writer spans were pre-query and never included QUERY, ANSWER, or the answer token.

Command:

- `python scripts/run_validation.py --steps 5 --batch-size 4 --d-model 64 --layers 1 --heads 4 --device cpu --eval-batches 2 --write-modes oracle,fact_token,random_write --task kv`

Stage 2 uses the original clean KV task only. No hard two-hop, distractor, learned marker scorer, top-k surprisal, or full write equation was added.

Controls: `normal`, `no_retrieval`, `shuffled_values`, `random_keys`.

## Summary Table

| write | model | seq_len | control | exact mean | exact std | CE mean | CE std | ret top1 | ret topk | slots | true write | false write | missed | ex/sec |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fact_token | epmem | 256 | no_retrieval | 0.0000 | 0.0000 | 61.3001 | 4.9789 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 390.16 |
| fact_token | epmem | 256 | normal | 0.7500 | 0.3853 | 1.9939 | 2.8295 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 412.42 |
| fact_token | epmem | 256 | random_keys | 0.2000 | 0.0685 | 38.0932 | 2.7915 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 382.64 |
| fact_token | epmem | 256 | shuffled_values | 0.0000 | 0.0000 | 48.0734 | 4.8288 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 392.94 |
| fact_token | epmem | 512 | no_retrieval | 0.0000 | 0.0000 | 60.3827 | 6.2726 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 189.99 |
| fact_token | epmem | 512 | normal | 0.6250 | 0.3307 | 1.6223 | 1.8584 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 191.69 |
| fact_token | epmem | 512 | random_keys | 0.1000 | 0.1046 | 40.6835 | 5.2931 | 0.2000 | 0.2000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 197.18 |
| fact_token | epmem | 512 | shuffled_values | 0.0000 | 0.0000 | 48.9933 | 2.3969 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 176.66 |
| fact_token | epmem | 1024 | no_retrieval | 0.0000 | 0.0000 | 59.8755 | 5.2381 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 58.10 |
| fact_token | epmem | 1024 | normal | 0.6750 | 0.1896 | 1.6793 | 1.6230 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 56.85 |
| fact_token | epmem | 1024 | random_keys | 0.1750 | 0.1425 | 37.5721 | 9.2642 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 56.92 |
| fact_token | epmem | 1024 | shuffled_values | 0.0000 | 0.0000 | 48.4607 | 4.8888 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 60.69 |
| fact_token | hebbian | 256 | no_retrieval | 0.0000 | 0.0000 | 62.0668 | 5.0216 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 428.36 |
| fact_token | hebbian | 256 | normal | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 385.87 |
| fact_token | hebbian | 256 | random_keys | 0.1000 | 0.0559 | 44.4010 | 7.1551 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 363.33 |
| fact_token | hebbian | 256 | shuffled_values | 0.0000 | 0.0000 | 60.4000 | 8.0862 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 363.14 |
| fact_token | hebbian | 512 | no_retrieval | 0.0000 | 0.0000 | 61.1595 | 6.3241 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 190.84 |
| fact_token | hebbian | 512 | normal | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 185.42 |
| fact_token | hebbian | 512 | random_keys | 0.0750 | 0.0685 | 50.7580 | 8.9711 | 0.2000 | 0.2000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 174.63 |
| fact_token | hebbian | 512 | shuffled_values | 0.0000 | 0.0000 | 62.2382 | 1.5525 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 176.82 |
| fact_token | hebbian | 1024 | no_retrieval | 0.0000 | 0.0000 | 60.6963 | 5.1928 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 59.12 |
| fact_token | hebbian | 1024 | normal | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 59.88 |
| fact_token | hebbian | 1024 | random_keys | 0.1500 | 0.1046 | 35.3386 | 12.5957 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 59.43 |
| fact_token | hebbian | 1024 | shuffled_values | 0.0000 | 0.0000 | 58.1903 | 6.4984 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 58.02 |
| fact_token | hpm_lite | 256 | no_retrieval | 0.0000 | 0.0000 | 61.1747 | 4.9940 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 397.21 |
| fact_token | hpm_lite | 256 | normal | 0.7500 | 0.3853 | 1.9894 | 2.8246 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 384.76 |
| fact_token | hpm_lite | 256 | random_keys | 0.2000 | 0.0685 | 38.1011 | 2.7821 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 377.22 |
| fact_token | hpm_lite | 256 | shuffled_values | 0.0000 | 0.0000 | 48.0528 | 4.8771 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 375.85 |
| fact_token | hpm_lite | 512 | no_retrieval | 0.0000 | 0.0000 | 60.1709 | 6.3297 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 193.24 |
| fact_token | hpm_lite | 512 | normal | 0.6250 | 0.3307 | 1.5742 | 1.8464 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 177.85 |
| fact_token | hpm_lite | 512 | random_keys | 0.1000 | 0.1046 | 40.5910 | 5.2256 | 0.2000 | 0.2000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 183.22 |
| fact_token | hpm_lite | 512 | shuffled_values | 0.0000 | 0.0000 | 48.8725 | 2.4294 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 170.52 |
| fact_token | hpm_lite | 1024 | no_retrieval | 0.0000 | 0.0000 | 59.5667 | 5.2040 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 57.29 |
| fact_token | hpm_lite | 1024 | normal | 0.6750 | 0.1896 | 1.5880 | 1.5667 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 60.94 |
| fact_token | hpm_lite | 1024 | random_keys | 0.1750 | 0.1425 | 37.4562 | 9.1355 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 55.14 |
| fact_token | hpm_lite | 1024 | shuffled_values | 0.0000 | 0.0000 | 48.3265 | 4.8479 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 59.67 |
| fact_token | local | 256 | no_retrieval | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 433.01 |
| fact_token | local | 256 | normal | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 410.94 |
| fact_token | local | 256 | random_keys | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 436.53 |
| fact_token | local | 256 | shuffled_values | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 403.92 |
| fact_token | local | 512 | no_retrieval | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 194.25 |
| fact_token | local | 512 | normal | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 190.12 |
| fact_token | local | 512 | random_keys | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 187.89 |
| fact_token | local | 512 | shuffled_values | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 186.32 |
| fact_token | local | 1024 | no_retrieval | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 59.65 |
| fact_token | local | 1024 | normal | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 60.62 |
| fact_token | local | 1024 | random_keys | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 59.59 |
| fact_token | local | 1024 | shuffled_values | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 61.06 |
| oracle | epmem | 256 | no_retrieval | 0.0000 | 0.0000 | 61.3001 | 4.9789 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 913.51 |
| oracle | epmem | 256 | normal | 0.7500 | 0.3853 | 1.9939 | 2.8295 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 862.89 |
| oracle | epmem | 256 | random_keys | 0.2000 | 0.0685 | 38.0932 | 2.7915 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 884.65 |
| oracle | epmem | 256 | shuffled_values | 0.0000 | 0.0000 | 48.0734 | 4.8288 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 854.26 |
| oracle | epmem | 512 | no_retrieval | 0.0000 | 0.0000 | 60.3827 | 6.2726 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 364.95 |
| oracle | epmem | 512 | normal | 0.6250 | 0.3307 | 1.6223 | 1.8584 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 342.37 |
| oracle | epmem | 512 | random_keys | 0.1000 | 0.1046 | 40.6835 | 5.2931 | 0.2000 | 0.2000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 339.66 |
| oracle | epmem | 512 | shuffled_values | 0.0000 | 0.0000 | 48.9933 | 2.3969 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 341.36 |
| oracle | epmem | 1024 | no_retrieval | 0.0000 | 0.0000 | 59.8755 | 5.2381 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 79.15 |
| oracle | epmem | 1024 | normal | 0.6750 | 0.1896 | 1.6793 | 1.6230 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 79.86 |
| oracle | epmem | 1024 | random_keys | 0.1750 | 0.1425 | 37.5721 | 9.2642 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 77.09 |
| oracle | epmem | 1024 | shuffled_values | 0.0000 | 0.0000 | 48.4607 | 4.8888 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 74.67 |
| oracle | hebbian | 256 | no_retrieval | 0.0000 | 0.0000 | 62.0668 | 5.0216 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 921.20 |
| oracle | hebbian | 256 | normal | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 746.40 |
| oracle | hebbian | 256 | random_keys | 0.1000 | 0.0559 | 44.4010 | 7.1551 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 762.08 |
| oracle | hebbian | 256 | shuffled_values | 0.0000 | 0.0000 | 60.4000 | 8.0862 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 745.39 |
| oracle | hebbian | 512 | no_retrieval | 0.0000 | 0.0000 | 61.1595 | 6.3241 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 353.13 |
| oracle | hebbian | 512 | normal | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 328.53 |
| oracle | hebbian | 512 | random_keys | 0.0750 | 0.0685 | 50.7580 | 8.9711 | 0.2000 | 0.2000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 322.29 |
| oracle | hebbian | 512 | shuffled_values | 0.0000 | 0.0000 | 62.2382 | 1.5525 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 325.46 |
| oracle | hebbian | 1024 | no_retrieval | 0.0000 | 0.0000 | 60.6963 | 5.1928 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 83.99 |
| oracle | hebbian | 1024 | normal | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 79.46 |
| oracle | hebbian | 1024 | random_keys | 0.1500 | 0.1046 | 35.3386 | 12.5957 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 83.73 |
| oracle | hebbian | 1024 | shuffled_values | 0.0000 | 0.0000 | 58.1903 | 6.4984 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 86.20 |
| oracle | hpm_lite | 256 | no_retrieval | 0.0000 | 0.0000 | 61.1747 | 4.9940 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 882.93 |
| oracle | hpm_lite | 256 | normal | 0.7500 | 0.3853 | 1.9894 | 2.8246 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 745.90 |
| oracle | hpm_lite | 256 | random_keys | 0.2000 | 0.0685 | 38.1011 | 2.7821 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 773.82 |
| oracle | hpm_lite | 256 | shuffled_values | 0.0000 | 0.0000 | 48.0528 | 4.8771 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 795.15 |
| oracle | hpm_lite | 512 | no_retrieval | 0.0000 | 0.0000 | 60.1709 | 6.3297 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 344.67 |
| oracle | hpm_lite | 512 | normal | 0.6250 | 0.3307 | 1.5742 | 1.8464 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 319.44 |
| oracle | hpm_lite | 512 | random_keys | 0.1000 | 0.1046 | 40.5910 | 5.2256 | 0.2000 | 0.2000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 320.07 |
| oracle | hpm_lite | 512 | shuffled_values | 0.0000 | 0.0000 | 48.8725 | 2.4294 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 322.87 |
| oracle | hpm_lite | 1024 | no_retrieval | 0.0000 | 0.0000 | 59.5667 | 5.2040 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 74.76 |
| oracle | hpm_lite | 1024 | normal | 0.6750 | 0.1896 | 1.5880 | 1.5667 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 71.06 |
| oracle | hpm_lite | 1024 | random_keys | 0.1750 | 0.1425 | 37.4562 | 9.1355 | 0.2250 | 0.2250 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 76.80 |
| oracle | hpm_lite | 1024 | shuffled_values | 0.0000 | 0.0000 | 48.3265 | 4.8479 | 1.0000 | 1.0000 | 4.00 | 1.0000 | 0.0000 | 0.0000 | 76.43 |
| oracle | local | 256 | no_retrieval | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 951.43 |
| oracle | local | 256 | normal | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 942.20 |
| oracle | local | 256 | random_keys | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 1007.87 |
| oracle | local | 256 | shuffled_values | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 994.13 |
| oracle | local | 512 | no_retrieval | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 368.26 |
| oracle | local | 512 | normal | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 370.32 |
| oracle | local | 512 | random_keys | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 365.37 |
| oracle | local | 512 | shuffled_values | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 366.85 |
| oracle | local | 1024 | no_retrieval | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 85.89 |
| oracle | local | 1024 | normal | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 84.56 |
| oracle | local | 1024 | random_keys | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 85.31 |
| oracle | local | 1024 | shuffled_values | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 1.0000 | 0.0000 | 0.0000 | 85.00 |
| random_write | epmem | 256 | no_retrieval | 0.0000 | 0.0000 | 61.2946 | 4.8947 |  |  | 4.00 | 0.0250 | 0.9750 | 0.9750 | 161.92 |
| random_write | epmem | 256 | normal | 0.0000 | 0.0000 | 47.2818 | 4.5870 | 1.0000 | 1.0000 | 4.00 | 0.0250 | 0.9750 | 0.9750 | 155.73 |
| random_write | epmem | 256 | random_keys | 0.0000 | 0.0000 | 48.1800 | 3.2399 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 159.59 |
| random_write | epmem | 256 | shuffled_values | 0.0000 | 0.0000 | 48.9492 | 2.9556 | 1.0000 | 1.0000 | 4.00 | 0.0250 | 0.9750 | 0.9750 | 160.80 |
| random_write | epmem | 512 | no_retrieval | 0.0000 | 0.0000 | 60.4165 | 6.2371 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 79.75 |
| random_write | epmem | 512 | normal | 0.0000 | 0.0000 | 47.7915 | 3.2821 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 84.20 |
| random_write | epmem | 512 | random_keys | 0.0000 | 0.0000 | 46.8116 | 2.5309 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 85.46 |
| random_write | epmem | 512 | shuffled_values | 0.0000 | 0.0000 | 47.6534 | 5.0883 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 82.52 |
| random_write | epmem | 1024 | no_retrieval | 0.0000 | 0.0000 | 60.1858 | 5.0939 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 29.98 |
| random_write | epmem | 1024 | normal | 0.0000 | 0.0000 | 46.9951 | 3.3681 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 31.87 |
| random_write | epmem | 1024 | random_keys | 0.0000 | 0.0000 | 46.3547 | 1.5055 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 29.95 |
| random_write | epmem | 1024 | shuffled_values | 0.0000 | 0.0000 | 46.3342 | 3.7977 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 29.44 |
| random_write | hebbian | 256 | no_retrieval | 0.0000 | 0.0000 | 61.0831 | 5.2123 |  |  | 4.00 | 0.0250 | 0.9750 | 0.9750 | 156.81 |
| random_write | hebbian | 256 | normal | 0.0250 | 0.0559 | 39.6077 | 4.9654 | 1.0000 | 1.0000 | 4.00 | 0.0250 | 0.9750 | 0.9750 | 155.39 |
| random_write | hebbian | 256 | random_keys | 0.0000 | 0.0000 | 42.1522 | 1.4254 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 163.98 |
| random_write | hebbian | 256 | shuffled_values | 0.0000 | 0.0000 | 40.8877 | 3.2461 | 1.0000 | 1.0000 | 4.00 | 0.0250 | 0.9750 | 0.9750 | 158.85 |
| random_write | hebbian | 512 | no_retrieval | 0.0000 | 0.0000 | 60.4638 | 6.3034 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 77.41 |
| random_write | hebbian | 512 | normal | 0.0000 | 0.0000 | 41.3617 | 4.3771 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 81.09 |
| random_write | hebbian | 512 | random_keys | 0.0000 | 0.0000 | 38.2918 | 5.6645 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 82.43 |
| random_write | hebbian | 512 | shuffled_values | 0.0000 | 0.0000 | 38.9383 | 3.2926 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 80.83 |
| random_write | hebbian | 1024 | no_retrieval | 0.0000 | 0.0000 | 59.8911 | 5.3563 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 32.11 |
| random_write | hebbian | 1024 | normal | 0.0000 | 0.0000 | 39.9532 | 3.5348 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 31.99 |
| random_write | hebbian | 1024 | random_keys | 0.0000 | 0.0000 | 40.0113 | 5.6823 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 30.61 |
| random_write | hebbian | 1024 | shuffled_values | 0.0000 | 0.0000 | 40.7487 | 3.1006 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 32.30 |
| random_write | hpm_lite | 256 | no_retrieval | 0.0000 | 0.0000 | 61.1117 | 4.7618 |  |  | 4.00 | 0.0250 | 0.9750 | 0.9750 | 161.13 |
| random_write | hpm_lite | 256 | normal | 0.0000 | 0.0000 | 46.9479 | 4.3907 | 1.0000 | 1.0000 | 4.00 | 0.0250 | 0.9750 | 0.9750 | 162.31 |
| random_write | hpm_lite | 256 | random_keys | 0.0000 | 0.0000 | 47.8938 | 3.1902 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 156.28 |
| random_write | hpm_lite | 256 | shuffled_values | 0.0000 | 0.0000 | 48.7678 | 3.0577 | 1.0000 | 1.0000 | 4.00 | 0.0250 | 0.9750 | 0.9750 | 161.75 |
| random_write | hpm_lite | 512 | no_retrieval | 0.0000 | 0.0000 | 60.3323 | 6.1830 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 78.76 |
| random_write | hpm_lite | 512 | normal | 0.0000 | 0.0000 | 47.6657 | 3.3762 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 80.42 |
| random_write | hpm_lite | 512 | random_keys | 0.0000 | 0.0000 | 46.7304 | 2.7194 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 77.78 |
| random_write | hpm_lite | 512 | shuffled_values | 0.0000 | 0.0000 | 47.6154 | 5.0462 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 80.46 |
| random_write | hpm_lite | 1024 | no_retrieval | 0.0000 | 0.0000 | 59.6975 | 5.0662 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 31.98 |
| random_write | hpm_lite | 1024 | normal | 0.0000 | 0.0000 | 46.7265 | 3.3608 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 33.50 |
| random_write | hpm_lite | 1024 | random_keys | 0.0000 | 0.0000 | 46.2260 | 1.3738 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 31.99 |
| random_write | hpm_lite | 1024 | shuffled_values | 0.0000 | 0.0000 | 46.2623 | 3.7893 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 31.35 |
| random_write | local | 256 | no_retrieval | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 0.0250 | 0.9750 | 0.9750 | 168.32 |
| random_write | local | 256 | normal | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 0.0250 | 0.9750 | 0.9750 | 160.69 |
| random_write | local | 256 | random_keys | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 0.0250 | 0.9750 | 0.9750 | 163.76 |
| random_write | local | 256 | shuffled_values | 0.0000 | 0.0000 | 60.4359 | 5.3060 |  |  | 4.00 | 0.0250 | 0.9750 | 0.9750 | 166.28 |
| random_write | local | 512 | no_retrieval | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 79.54 |
| random_write | local | 512 | normal | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 76.63 |
| random_write | local | 512 | random_keys | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 80.12 |
| random_write | local | 512 | shuffled_values | 0.0000 | 0.0000 | 59.6574 | 6.6115 |  |  | 4.00 | 0.0063 | 0.9938 | 0.9938 | 80.97 |
| random_write | local | 1024 | no_retrieval | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 32.77 |
| random_write | local | 1024 | normal | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 31.05 |
| random_write | local | 1024 | random_keys | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 30.88 |
| random_write | local | 1024 | shuffled_values | 0.0000 | 0.0000 | 59.4631 | 5.3544 |  |  | 4.00 | 0.0000 | 1.0000 | 1.0000 | 31.32 |

## Stage 2 Verdict

Oracle retrieval sanity: epmem normal exact averaged 0.683. Fact-token writer: epmem averaged 0.683, a 0.0 point delta from oracle. Random-write control: epmem averaged 0.000, 68.3 points below fact-token. Hebbian without oracle metadata stayed at 1.000 versus oracle 1.000. HPM-Lite versus epmem under fact-token writing: 0.683 vs 0.683. HPM-Lite no-retrieval under fact-token writing averaged 0.000. This is clean KV only; no hard two-hop or distractor conclusions should be drawn from this stage.

Expected checks: fact-token should track oracle on clean KV; random-write, no-retrieval, shuffled-values, and random-keys should be worse. If random-write performs well, inspect leakage or shortcuts.

## Known Limitations

- Stage 2 is clean KV only.
- The run is intentionally small enough for CPU validation.
- Fact-token writing is a parser baseline, not a learned writer.

## MEMFAIL-Lite Diagnostics

This section refactors the plan from a single clean recall task toward small diagnostic tasks that separate memory failure modes. It keeps the existing Stage 2 KV results above intact and does not add a learned writer, JEPA, ANN, Priming, GKA, or full HPM routing.

Tasks: `kv,coexisting,conditional,longhop`. Models: `local,epmem,hpm_lite,hebbian`. Write modes: `oracle,fact_token,random_write`. Controls: `normal,no_retrieval,shuffled_values,random_keys`.
Seeds: `0,1,2,3,4`. Sequence lengths: `512`. Window: `64`.
Steps: `3`. Batch size: `4`. Eval batches: `1`. Top-k: `2`.
Device request: `cpu`. Platform: `Windows-10-10.0.19045-SP0`. Torch: `2.12.0+cpu`.
Leak checks passed on `160` generated examples across all requested write modes; memory writes stayed pre-query and excluded future answer tokens.

Command:

- `python scripts/run_memfail_lite.py --steps 3 --batch-size 4 --d-model 64 --layers 1 --heads 4 --device cpu --eval-batches 1 --seeds 0,1,2,3,4 --seq-lens 512`

Raw and summarized outputs: `runs/memfail_raw.csv`, `runs/memfail_summary.csv`.

### Compact Summary

| task | write | model | seq_len | control | exact | CE | ret top1 | ret topk | use if ret | slots | true write | false write | missed | ex/sec |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coexisting | fact_token | epmem | 512 | no_retrieval | 0.0000 | 60.7796 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 186.3947 |
| coexisting | fact_token | epmem | 512 | normal | 0.0000 | 42.6304 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 169.3523 |
| coexisting | fact_token | epmem | 512 | random_keys | 0.0000 | 48.3233 | 0.6000 | 0.1000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 175.8636 |
| coexisting | fact_token | epmem | 512 | shuffled_values | 0.0000 | 49.2062 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 165.5440 |
| coexisting | fact_token | hebbian | 512 | no_retrieval | 0.0000 | 60.9846 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 174.3086 |
| coexisting | fact_token | hebbian | 512 | normal | 0.0000 | 34.7143 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 162.4033 |
| coexisting | fact_token | hebbian | 512 | random_keys | 0.0000 | 51.6760 | 0.6000 | 0.1000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 173.7676 |
| coexisting | fact_token | hebbian | 512 | shuffled_values | 0.0000 | 47.4521 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 164.1950 |
| coexisting | fact_token | hpm_lite | 512 | no_retrieval | 0.0000 | 60.7194 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 175.0447 |
| coexisting | fact_token | hpm_lite | 512 | normal | 0.0000 | 42.5689 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 151.7895 |
| coexisting | fact_token | hpm_lite | 512 | random_keys | 0.0000 | 48.3949 | 0.5500 | 0.3000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 161.5442 |
| coexisting | fact_token | hpm_lite | 512 | shuffled_values | 0.0000 | 49.1574 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 158.0274 |
| coexisting | fact_token | local | 512 | no_retrieval | 0.0000 | 60.6681 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 175.3604 |
| coexisting | fact_token | local | 512 | normal | 0.0000 | 60.6681 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 173.1801 |
| coexisting | fact_token | local | 512 | random_keys | 0.0000 | 60.6681 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 190.7908 |
| coexisting | fact_token | local | 512 | shuffled_values | 0.0000 | 60.6681 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 175.8920 |
| coexisting | oracle | epmem | 512 | no_retrieval | 0.0000 | 60.7796 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 307.0681 |
| coexisting | oracle | epmem | 512 | normal | 0.0000 | 42.6304 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 287.2021 |
| coexisting | oracle | epmem | 512 | random_keys | 0.0000 | 48.3233 | 0.6000 | 0.1000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 293.8072 |
| coexisting | oracle | epmem | 512 | shuffled_values | 0.0000 | 49.2062 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 284.6666 |
| coexisting | oracle | hebbian | 512 | no_retrieval | 0.0000 | 60.9846 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 288.7576 |
| coexisting | oracle | hebbian | 512 | normal | 0.0000 | 34.7143 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 242.5561 |
| coexisting | oracle | hebbian | 512 | random_keys | 0.0000 | 51.6760 | 0.6000 | 0.1000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 275.5350 |
| coexisting | oracle | hebbian | 512 | shuffled_values | 0.0000 | 47.4521 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 280.5498 |
| coexisting | oracle | hpm_lite | 512 | no_retrieval | 0.0000 | 60.7194 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 292.1421 |
| coexisting | oracle | hpm_lite | 512 | normal | 0.0000 | 42.5689 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 262.2429 |
| coexisting | oracle | hpm_lite | 512 | random_keys | 0.0000 | 48.3949 | 0.5500 | 0.3000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 276.7725 |
| coexisting | oracle | hpm_lite | 512 | shuffled_values | 0.0000 | 49.1574 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 270.9203 |
| coexisting | oracle | local | 512 | no_retrieval | 0.0000 | 60.6681 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 311.4921 |
| coexisting | oracle | local | 512 | normal | 0.0000 | 60.6681 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 308.3348 |
| coexisting | oracle | local | 512 | random_keys | 0.0000 | 60.6681 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 310.0493 |
| coexisting | oracle | local | 512 | shuffled_values | 0.0000 | 60.6681 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 315.7078 |
| coexisting | random_write | epmem | 512 | no_retrieval | 0.0000 | 60.7665 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 80.0667 |
| coexisting | random_write | epmem | 512 | normal | 0.0000 | 54.9928 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 0.0250 | 0.9750 | 0.9750 | 78.9529 |
| coexisting | random_write | epmem | 512 | random_keys | 0.0000 | 54.2773 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 75.9408 |
| coexisting | random_write | epmem | 512 | shuffled_values | 0.0000 | 55.9098 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 78.5327 |
| coexisting | random_write | hebbian | 512 | no_retrieval | 0.0000 | 60.8688 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 76.3517 |
| coexisting | random_write | hebbian | 512 | normal | 0.0000 | 51.8528 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 0.0250 | 0.9750 | 0.9750 | 77.0365 |
| coexisting | random_write | hebbian | 512 | random_keys | 0.0000 | 50.6239 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 73.8422 |
| coexisting | random_write | hebbian | 512 | shuffled_values | 0.0000 | 52.8303 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 81.3576 |
| coexisting | random_write | hpm_lite | 512 | no_retrieval | 0.0000 | 60.6763 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 78.0716 |
| coexisting | random_write | hpm_lite | 512 | normal | 0.0000 | 55.0486 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 80.7372 |
| coexisting | random_write | hpm_lite | 512 | random_keys | 0.0000 | 54.3535 | 0.0000 | 1.0000 | 0.0000 | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 74.1013 |
| coexisting | random_write | hpm_lite | 512 | shuffled_values | 0.0000 | 55.3160 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 77.4537 |
| coexisting | random_write | local | 512 | no_retrieval | 0.0000 | 60.6681 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 79.5272 |
| coexisting | random_write | local | 512 | normal | 0.0000 | 60.6681 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 77.8449 |
| coexisting | random_write | local | 512 | random_keys | 0.0000 | 60.6681 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 78.8059 |
| coexisting | random_write | local | 512 | shuffled_values | 0.0000 | 60.6681 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 79.9660 |
| conditional | fact_token | epmem | 512 | no_retrieval | 0.0000 | 56.1493 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 160.8170 |
| conditional | fact_token | epmem | 512 | normal | 0.1000 | 25.9605 | 0.4333 | 0.9333 | 0.2000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 162.9470 |
| conditional | fact_token | epmem | 512 | random_keys | 0.0500 | 36.0537 | 0.1500 | 0.6500 | 0.2500 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 164.5728 |
| conditional | fact_token | epmem | 512 | shuffled_values | 0.0000 | 41.3110 | 0.4333 | 0.9333 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 170.7185 |
| conditional | fact_token | hebbian | 512 | no_retrieval | 0.0000 | 56.3832 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 170.8411 |
| conditional | fact_token | hebbian | 512 | normal | 0.3000 | 17.5294 | 0.4333 | 0.9333 | 0.5333 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 154.7078 |
| conditional | fact_token | hebbian | 512 | random_keys | 0.1000 | 37.9840 | 0.1500 | 0.6500 | 0.3333 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 178.3267 |
| conditional | fact_token | hebbian | 512 | shuffled_values | 0.0500 | 42.6205 | 0.4333 | 0.9333 | 0.2000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 155.7779 |
| conditional | fact_token | hpm_lite | 512 | no_retrieval | 0.0000 | 55.9071 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 159.7086 |
| conditional | fact_token | hpm_lite | 512 | normal | 0.1000 | 25.7348 | 0.4333 | 0.9333 | 0.2000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 138.3901 |
| conditional | fact_token | hpm_lite | 512 | random_keys | 0.0500 | 38.8114 | 0.1500 | 0.4667 | 0.2500 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 148.9714 |
| conditional | fact_token | hpm_lite | 512 | shuffled_values | 0.0000 | 41.0397 | 0.4333 | 0.9333 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 153.1512 |
| conditional | fact_token | local | 512 | no_retrieval | 0.0000 | 55.7498 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 156.6327 |
| conditional | fact_token | local | 512 | normal | 0.0000 | 55.7498 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 154.9097 |
| conditional | fact_token | local | 512 | random_keys | 0.0000 | 55.7498 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 161.5469 |
| conditional | fact_token | local | 512 | shuffled_values | 0.0000 | 55.7498 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 156.2580 |
| conditional | oracle | epmem | 512 | no_retrieval | 0.0000 | 56.1493 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 283.4925 |
| conditional | oracle | epmem | 512 | normal | 0.1000 | 25.9605 | 0.4333 | 0.9333 | 0.2000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 268.8530 |
| conditional | oracle | epmem | 512 | random_keys | 0.0500 | 36.0537 | 0.1500 | 0.6500 | 0.2500 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 278.6599 |
| conditional | oracle | epmem | 512 | shuffled_values | 0.0000 | 41.3110 | 0.4333 | 0.9333 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 267.7926 |
| conditional | oracle | hebbian | 512 | no_retrieval | 0.0000 | 56.3832 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 292.5716 |
| conditional | oracle | hebbian | 512 | normal | 0.3000 | 17.5294 | 0.4333 | 0.9333 | 0.5333 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 264.0805 |
| conditional | oracle | hebbian | 512 | random_keys | 0.1000 | 37.9840 | 0.1500 | 0.6500 | 0.3333 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 273.2876 |
| conditional | oracle | hebbian | 512 | shuffled_values | 0.0500 | 42.6205 | 0.4333 | 0.9333 | 0.2000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 277.5456 |
| conditional | oracle | hpm_lite | 512 | no_retrieval | 0.0000 | 55.9071 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 269.2826 |
| conditional | oracle | hpm_lite | 512 | normal | 0.1000 | 25.7348 | 0.4333 | 0.9333 | 0.2000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 264.6149 |
| conditional | oracle | hpm_lite | 512 | random_keys | 0.0500 | 38.8114 | 0.1500 | 0.4667 | 0.2500 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 255.8204 |
| conditional | oracle | hpm_lite | 512 | shuffled_values | 0.0000 | 41.0397 | 0.4333 | 0.9333 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 259.8800 |
| conditional | oracle | local | 512 | no_retrieval | 0.0000 | 55.7498 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 300.6765 |
| conditional | oracle | local | 512 | normal | 0.0000 | 55.7498 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 292.1881 |
| conditional | oracle | local | 512 | random_keys | 0.0000 | 55.7498 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 299.3370 |
| conditional | oracle | local | 512 | shuffled_values | 0.0000 | 55.7498 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 293.6927 |
| conditional | random_write | epmem | 512 | no_retrieval | 0.0000 | 56.0209 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 77.5250 |
| conditional | random_write | epmem | 512 | normal | 0.0000 | 45.7044 |  |  |  | 4.0000 | 0.0250 | 0.9750 | 0.9750 | 78.0241 |
| conditional | random_write | epmem | 512 | random_keys | 0.0000 | 47.9807 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 79.2843 |
| conditional | random_write | epmem | 512 | shuffled_values | 0.0000 | 45.0039 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 79.7657 |
| conditional | random_write | hebbian | 512 | no_retrieval | 0.0000 | 56.1101 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 76.9795 |
| conditional | random_write | hebbian | 512 | normal | 0.0000 | 37.4798 |  |  |  | 4.0000 | 0.0250 | 0.9750 | 0.9750 | 76.7969 |
| conditional | random_write | hebbian | 512 | random_keys | 0.0000 | 43.8350 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 75.7986 |
| conditional | random_write | hebbian | 512 | shuffled_values | 0.0000 | 40.1948 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 76.2505 |
| conditional | random_write | hpm_lite | 512 | no_retrieval | 0.0000 | 55.7678 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 75.5980 |
| conditional | random_write | hpm_lite | 512 | normal | 0.0000 | 45.8249 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 77.1134 |
| conditional | random_write | hpm_lite | 512 | random_keys | 0.0000 | 44.9815 | 0.0000 | 0.0000 |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 73.4125 |
| conditional | random_write | hpm_lite | 512 | shuffled_values | 0.0000 | 44.8898 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 77.1771 |
| conditional | random_write | local | 512 | no_retrieval | 0.0000 | 55.7498 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 84.5485 |
| conditional | random_write | local | 512 | normal | 0.0000 | 55.7498 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 74.5008 |
| conditional | random_write | local | 512 | random_keys | 0.0000 | 55.7498 |  |  |  | 4.0000 | 0.0250 | 0.9750 | 0.9750 | 77.1862 |
| conditional | random_write | local | 512 | shuffled_values | 0.0000 | 55.7498 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 78.0841 |
| kv | fact_token | epmem | 512 | no_retrieval | 0.0000 | 60.3139 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 162.6352 |
| kv | fact_token | epmem | 512 | normal | 0.5000 | 3.4661 | 1.0000 | 1.0000 | 0.5000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 152.2876 |
| kv | fact_token | epmem | 512 | random_keys | 0.0000 | 37.9438 | 0.3000 | 0.4000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 166.6315 |
| kv | fact_token | epmem | 512 | shuffled_values | 0.0000 | 48.5696 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 150.0167 |
| kv | fact_token | hebbian | 512 | no_retrieval | 0.0000 | 60.5979 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 164.3662 |
| kv | fact_token | hebbian | 512 | normal | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 150.4774 |
| kv | fact_token | hebbian | 512 | random_keys | 0.1500 | 41.5852 | 0.3000 | 0.4000 | 0.3750 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 167.8050 |
| kv | fact_token | hebbian | 512 | shuffled_values | 0.0000 | 64.0179 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 156.2197 |
| kv | fact_token | hpm_lite | 512 | no_retrieval | 0.0000 | 60.1749 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 165.2683 |
| kv | fact_token | hpm_lite | 512 | normal | 0.5000 | 3.4429 | 1.0000 | 1.0000 | 0.5000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 149.9106 |
| kv | fact_token | hpm_lite | 512 | random_keys | 0.0500 | 31.2168 | 0.3000 | 0.7500 | 0.0667 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 160.4543 |
| kv | fact_token | hpm_lite | 512 | shuffled_values | 0.0000 | 48.4864 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 162.5808 |
| kv | fact_token | local | 512 | no_retrieval | 0.0000 | 59.8721 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 163.1240 |
| kv | fact_token | local | 512 | normal | 0.0000 | 59.8721 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 163.7671 |
| kv | fact_token | local | 512 | random_keys | 0.0000 | 59.8721 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 159.7106 |
| kv | fact_token | local | 512 | shuffled_values | 0.0000 | 59.8721 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 161.8597 |
| kv | oracle | epmem | 512 | no_retrieval | 0.0000 | 60.3139 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 286.8266 |
| kv | oracle | epmem | 512 | normal | 0.5000 | 3.4661 | 1.0000 | 1.0000 | 0.5000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 254.6899 |
| kv | oracle | epmem | 512 | random_keys | 0.0000 | 37.9438 | 0.3000 | 0.4000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 271.6752 |
| kv | oracle | epmem | 512 | shuffled_values | 0.0000 | 48.5696 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 264.3403 |
| kv | oracle | hebbian | 512 | no_retrieval | 0.0000 | 60.5979 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 287.4840 |
| kv | oracle | hebbian | 512 | normal | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 258.4668 |
| kv | oracle | hebbian | 512 | random_keys | 0.1500 | 41.5852 | 0.3000 | 0.4000 | 0.3750 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 268.7629 |
| kv | oracle | hebbian | 512 | shuffled_values | 0.0000 | 64.0179 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 254.7208 |
| kv | oracle | hpm_lite | 512 | no_retrieval | 0.0000 | 60.1749 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 267.6347 |
| kv | oracle | hpm_lite | 512 | normal | 0.5000 | 3.4429 | 1.0000 | 1.0000 | 0.5000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 260.2122 |
| kv | oracle | hpm_lite | 512 | random_keys | 0.0500 | 31.2168 | 0.3000 | 0.7500 | 0.0667 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 257.9174 |
| kv | oracle | hpm_lite | 512 | shuffled_values | 0.0000 | 48.4864 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 255.6788 |
| kv | oracle | local | 512 | no_retrieval | 0.0000 | 59.8721 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 284.3671 |
| kv | oracle | local | 512 | normal | 0.0000 | 59.8721 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 283.7570 |
| kv | oracle | local | 512 | random_keys | 0.0000 | 59.8721 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 277.2743 |
| kv | oracle | local | 512 | shuffled_values | 0.0000 | 59.8721 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 285.6958 |
| kv | random_write | epmem | 512 | no_retrieval | 0.0000 | 60.1610 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 75.9958 |
| kv | random_write | epmem | 512 | normal | 0.0000 | 49.8114 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 75.0633 |
| kv | random_write | epmem | 512 | random_keys | 0.0000 | 47.0100 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 74.9392 |
| kv | random_write | epmem | 512 | shuffled_values | 0.0000 | 48.3976 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 76.4162 |
| kv | random_write | hebbian | 512 | no_retrieval | 0.0000 | 60.2041 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 79.5184 |
| kv | random_write | hebbian | 512 | normal | 0.0000 | 42.0656 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 70.7708 |
| kv | random_write | hebbian | 512 | random_keys | 0.0000 | 39.5178 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 69.7704 |
| kv | random_write | hebbian | 512 | shuffled_values | 0.0000 | 40.7069 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 72.1284 |
| kv | random_write | hpm_lite | 512 | no_retrieval | 0.0000 | 59.9866 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 71.9047 |
| kv | random_write | hpm_lite | 512 | normal | 0.0000 | 50.7477 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 74.1362 |
| kv | random_write | hpm_lite | 512 | random_keys | 0.0000 | 48.5820 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 73.2226 |
| kv | random_write | hpm_lite | 512 | shuffled_values | 0.0000 | 46.6271 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 73.0537 |
| kv | random_write | local | 512 | no_retrieval | 0.0000 | 59.8721 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 72.5690 |
| kv | random_write | local | 512 | normal | 0.0000 | 59.8721 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 68.8200 |
| kv | random_write | local | 512 | random_keys | 0.0000 | 59.8721 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 72.3813 |
| kv | random_write | local | 512 | shuffled_values | 0.0000 | 59.8721 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 71.8750 |
| longhop | fact_token | epmem | 512 | no_retrieval | 0.0000 | 63.5385 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 163.9761 |
| longhop | fact_token | epmem | 512 | normal | 0.6000 | 2.3887 | 1.0000 | 1.0000 | 0.6000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 157.1367 |
| longhop | fact_token | epmem | 512 | random_keys | 0.0500 | 30.7726 | 0.5000 | 0.6500 | 0.0667 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 157.9419 |
| longhop | fact_token | epmem | 512 | shuffled_values | 0.0500 | 31.4838 | 0.1000 | 0.4500 | 0.0833 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 158.6806 |
| longhop | fact_token | hebbian | 512 | no_retrieval | 0.0000 | 63.6752 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 164.8403 |
| longhop | fact_token | hebbian | 512 | normal | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 153.9036 |
| longhop | fact_token | hebbian | 512 | random_keys | 0.3000 | 36.9580 | 0.2000 | 0.5500 | 0.6000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 156.8696 |
| longhop | fact_token | hebbian | 512 | shuffled_values | 0.2000 | 37.6820 | 0.4000 | 0.6000 | 0.1333 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 163.0577 |
| longhop | fact_token | hpm_lite | 512 | no_retrieval | 0.0000 | 63.5546 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 169.8595 |
| longhop | fact_token | hpm_lite | 512 | normal | 0.6500 | 2.3612 | 1.0000 | 1.0000 | 0.6500 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 158.0532 |
| longhop | fact_token | hpm_lite | 512 | random_keys | 0.0000 | 38.0531 | 0.2000 | 0.5500 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 163.9746 |
| longhop | fact_token | hpm_lite | 512 | shuffled_values | 0.0500 | 31.4796 | 0.1000 | 0.4500 | 0.0833 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 165.7039 |
| longhop | fact_token | local | 512 | no_retrieval | 0.0000 | 63.0893 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 170.8543 |
| longhop | fact_token | local | 512 | normal | 0.0000 | 63.0893 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 166.1414 |
| longhop | fact_token | local | 512 | random_keys | 0.0000 | 63.0893 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 172.9071 |
| longhop | fact_token | local | 512 | shuffled_values | 0.0000 | 63.0893 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 166.4745 |
| longhop | oracle | epmem | 512 | no_retrieval | 0.0000 | 63.5385 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 300.1017 |
| longhop | oracle | epmem | 512 | normal | 0.6000 | 2.3887 | 1.0000 | 1.0000 | 0.6000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 271.5658 |
| longhop | oracle | epmem | 512 | random_keys | 0.0500 | 30.7726 | 0.5000 | 0.6500 | 0.0667 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 273.0145 |
| longhop | oracle | epmem | 512 | shuffled_values | 0.0500 | 31.4838 | 0.1000 | 0.4500 | 0.0833 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 278.4285 |
| longhop | oracle | hebbian | 512 | no_retrieval | 0.0000 | 63.6752 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 295.0537 |
| longhop | oracle | hebbian | 512 | normal | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 266.3778 |
| longhop | oracle | hebbian | 512 | random_keys | 0.3000 | 36.9580 | 0.2000 | 0.5500 | 0.6000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 271.1922 |
| longhop | oracle | hebbian | 512 | shuffled_values | 0.2000 | 37.6820 | 0.4000 | 0.6000 | 0.1333 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 274.3940 |
| longhop | oracle | hpm_lite | 512 | no_retrieval | 0.0000 | 63.5546 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 267.8055 |
| longhop | oracle | hpm_lite | 512 | normal | 0.6500 | 2.3612 | 1.0000 | 1.0000 | 0.6500 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 256.0997 |
| longhop | oracle | hpm_lite | 512 | random_keys | 0.0000 | 38.0531 | 0.2000 | 0.5500 | 0.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 261.2503 |
| longhop | oracle | hpm_lite | 512 | shuffled_values | 0.0500 | 31.4796 | 0.1000 | 0.4500 | 0.0833 | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 255.3371 |
| longhop | oracle | local | 512 | no_retrieval | 0.0000 | 63.0893 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 305.3946 |
| longhop | oracle | local | 512 | normal | 0.0000 | 63.0893 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 291.4400 |
| longhop | oracle | local | 512 | random_keys | 0.0000 | 63.0893 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 301.9077 |
| longhop | oracle | local | 512 | shuffled_values | 0.0000 | 63.0893 |  |  |  | 4.0000 | 1.0000 | 0.0000 | 0.0000 | 298.3919 |
| longhop | random_write | epmem | 512 | no_retrieval | 0.0000 | 63.2350 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 72.0050 |
| longhop | random_write | epmem | 512 | normal | 0.0000 | 50.8832 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 77.7649 |
| longhop | random_write | epmem | 512 | random_keys | 0.0000 | 50.2176 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 74.7149 |
| longhop | random_write | epmem | 512 | shuffled_values | 0.0000 | 50.6689 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 77.3585 |
| longhop | random_write | hebbian | 512 | no_retrieval | 0.0000 | 63.4892 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 77.0633 |
| longhop | random_write | hebbian | 512 | normal | 0.0000 | 46.6559 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 73.3969 |
| longhop | random_write | hebbian | 512 | random_keys | 0.0000 | 41.2638 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 74.2547 |
| longhop | random_write | hebbian | 512 | shuffled_values | 0.0000 | 42.0583 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 74.6124 |
| longhop | random_write | hpm_lite | 512 | no_retrieval | 0.0000 | 63.2542 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 70.4059 |
| longhop | random_write | hpm_lite | 512 | normal | 0.0000 | 49.4476 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 73.0983 |
| longhop | random_write | hpm_lite | 512 | random_keys | 0.0000 | 52.1133 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 72.6108 |
| longhop | random_write | hpm_lite | 512 | shuffled_values | 0.0000 | 48.3797 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 70.8388 |
| longhop | random_write | local | 512 | no_retrieval | 0.0000 | 63.0893 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 79.5295 |
| longhop | random_write | local | 512 | normal | 0.0000 | 63.0893 |  |  |  | 4.0000 | 0.0000 | 1.0000 | 1.0000 | 76.9435 |
| longhop | random_write | local | 512 | random_keys | 0.0000 | 63.0893 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 79.7116 |
| longhop | random_write | local | 512 | shuffled_values | 0.0000 | 63.0893 |  |  |  | 4.0000 | 0.0125 | 0.9875 | 0.9875 | 78.5420 |

### MEMFAIL-Lite Verdict

- Exact recall: local fails (0.000); epmem is borderline (0.500); hpm_lite is borderline (0.500); hebbian passes (1.000).
- Coexisting facts: local fails (0.000); epmem fails (0.000); hpm_lite fails (0.000); hebbian fails (0.000).
- Conditional facts: local fails (0.000); epmem fails (0.100); hpm_lite fails (0.100); hebbian fails (0.300).
- Long-hop composition: local fails (0.000); epmem is borderline (0.600); hpm_lite is borderline (0.650); hebbian passes (1.000) on this clean two-hop setup.
- HPM-Lite vs epmem: largest fact-token normal delta was 5.0 points on `longhop`.
- Hebbian: clean KV exact was 1.000; the mean across interference/qualifier/composition tasks was 0.433. It still looks like a strong simple association baseline, not a robust memory system: it fails coexisting/conditional diagnostics here, and its clean long-hop success should not override the prior hard-audit failure under random order/corruption controls.
- This MEMFAIL-Lite run keeps distractor complexity off; failures here are diagnostic, not a full robustness audit.

Interpretation guardrail: the table above is a small CPU diagnostic run. It is meant to show whether the tasks and controls execute cleanly and expose separable failure modes, not to claim final model quality.

## Training Budget Sweep

This sweep checks whether MEMFAIL-Lite failures disappear with more optimization. It does not add learned writing, distractors, JEPA, ANN, GKA, Priming, RL, graph memory, or a new architecture.

Tasks: `kv,coexisting,conditional,longhop`. Models: `local,epmem,hpm_lite,hebbian`. Write modes: `oracle,fact_token`. Controls: `normal,no_retrieval,shuffled_values,random_keys`.
Budgets: `3,10,30,100,300`. Seeds: `0,1,2,3,4`. Seq len/window: `512` / `64`.
Model size: d_model `64`, layers `1`, heads `4`, batch size `4`.
Eval batches: `5`. Device request: `cuda`. Platform: `Windows-10-10.0.19045-SP0`. Torch: `2.11.0+cu128`.
Leak checks passed on `80` generated examples; memory writes stayed pre-query and excluded future answer tokens.

Command:

- `python scripts/run_memfail_budget.py --budgets 3,10,30,100,300 --seeds 0,1,2,3,4 --seq-len 512 --window 64 --batch-size 4 --eval-batches 5 --d-model 64 --layers 1 --heads 4 --device cuda`

Raw and summarized outputs: `runs/memfail_budget_raw.csv`, `runs/memfail_budget_summary.csv`.

### Fact-Token Normal Accuracy By Budget

| task | model | 3 | 10 | 30 | 100 | 300 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| kv | local | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0100 |
| kv | epmem | 0.6400 | 0.8600 | 1.0000 | 1.0000 | 1.0000 |
| kv | hpm_lite | 0.6600 | 0.8600 | 1.0000 | 1.0000 | 1.0000 |
| kv | hebbian | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| coexisting | local | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| coexisting | epmem | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0100 |
| coexisting | hpm_lite | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0100 |
| coexisting | hebbian | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| conditional | local | 0.0000 | 0.0000 | 0.1800 | 0.4200 | 0.4400 |
| conditional | epmem | 0.0200 | 0.0200 | 0.2700 | 0.4300 | 0.3400 |
| conditional | hpm_lite | 0.0200 | 0.0200 | 0.2800 | 0.4400 | 0.3400 |
| conditional | hebbian | 0.2700 | 0.2800 | 0.2400 | 0.4200 | 0.3100 |
| longhop | local | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| longhop | epmem | 0.7300 | 0.8600 | 1.0000 | 1.0000 | 1.0000 |
| longhop | hpm_lite | 0.7300 | 0.8500 | 1.0000 | 1.0000 | 1.0000 |
| longhop | hebbian | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Final-Budget Controls

| task | model | normal | no_retrieval | shuffled_values | random_keys |
| --- | --- | ---: | ---: | ---: | ---: |
| kv | local | 0.0100 | 0.0100 | 0.0100 | 0.0100 |
| kv | epmem | 1.0000 | 0.0000 | 0.0000 | 0.1900 |
| kv | hpm_lite | 1.0000 | 0.0000 | 0.0000 | 0.1900 |
| kv | hebbian | 1.0000 | 0.0000 | 0.0000 | 0.1000 |
| coexisting | local | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| coexisting | epmem | 0.0100 | 0.0000 | 0.0100 | 0.0100 |
| coexisting | hpm_lite | 0.0100 | 0.0000 | 0.0100 | 0.0100 |
| coexisting | hebbian | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| conditional | local | 0.4400 | 0.4400 | 0.4400 | 0.4400 |
| conditional | epmem | 0.3400 | 0.4400 | 0.3300 | 0.2300 |
| conditional | hpm_lite | 0.3400 | 0.4400 | 0.3000 | 0.2300 |
| conditional | hebbian | 0.3100 | 0.4400 | 0.2800 | 0.4400 |
| longhop | local | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| longhop | epmem | 1.0000 | 0.0000 | 0.2900 | 0.1800 |
| longhop | hpm_lite | 1.0000 | 0.0000 | 0.2800 | 0.1700 |
| longhop | hebbian | 1.0000 | 0.0000 | 0.1800 | 0.1500 |

### Budget Verdict

- Coexisting at 300 steps: epmem top-k 1.000, exact 0.010; mark as read/use composition failure.
- Conditional at 300 steps: epmem top-k 1.000, exact 0.340; mark as qualifier-use failure.
- HPM-Lite is not justified yet: strongest observed delta was 2.0 points on `kv` at 3 steps, not a consistent 5-10 point win.
- Hebbian at final budget: kv 1.000, coexisting 0.000, conditional 0.310, longhop 1.000.
- No-retrieval control at final budget averaged 0.111 exact.
- Controls still hurt: shuffled-values mean drop 44.4 points, random-keys mean drop 42.6 points at final budget.
- If any failure vanishes at larger budgets, treat it as insufficient training rather than an architecture failure.

## Read/Use Diagnostics

This run diagnoses whether the coexisting and conditional failures are memory failures or read/use failures. It keeps the task generators, writer, and memory architecture unchanged.

Tasks: `coexisting,conditional`. Models: `local,epmem,hpm_lite,hebbian`. Write modes: `oracle,fact_token`. Controls: `normal,no_retrieval,shuffled_values,random_keys`.
Budgets: `30,100,300`. Seeds: `0,1,2,3,4`. Seq len/window: `512` / `64`. Top-k: `4`.
Device request: `cuda`. Platform: `Windows-10-10.0.19045-SP0`. Torch: `2.11.0+cu128`.
Leak checks passed on `40` generated examples.

Command:

- `python scripts/run_readuse_diagnostics.py --budgets 30,100,300 --seeds 0,1,2,3,4 --seq-len 512 --window 64 --batch-size 4 --eval-batches 5 --d-model 64 --layers 1 --heads 4 --device cuda`

Raw and summarized outputs: `runs/readuse_raw.csv`, `runs/readuse_summary.csv`.

### Coexisting: CE Decoder vs Structured Readout

| model | budget | CE exact | retrieval top-k | structured set exact | per-value F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| local | 30 | 0.0000 |  | 0.0000 | 0.0000 |
| local | 100 | 0.0000 |  | 0.0000 | 0.0000 |
| local | 300 | 0.0000 |  | 0.0000 | 0.0000 |
| epmem | 30 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| epmem | 100 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| epmem | 300 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hpm_lite | 30 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hpm_lite | 100 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hpm_lite | 300 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hebbian | 30 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hebbian | 100 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hebbian | 300 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

### Conditional Split Metrics

| model | budget | exact | positive acc | negative acc | no-value bias | target no-value | pred value | retrieval top-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local | 30 | 0.1800 | 0.0000 | 0.4000 | 0.4000 | 0.5200 | 0.6000 |  |
| local | 100 | 0.5400 | 0.0000 | 1.0000 | 1.0000 | 0.5400 | 0.0000 |  |
| local | 300 | 0.4800 | 0.0000 | 1.0000 | 1.0000 | 0.4800 | 0.0000 |  |
| epmem | 30 | 0.2700 | 0.3047 | 0.2826 | 0.2700 | 0.5200 | 0.7300 | 1.0000 |
| epmem | 100 | 0.4400 | 0.2587 | 0.6140 | 0.5700 | 0.5400 | 0.4300 | 1.0000 |
| epmem | 300 | 0.2900 | 0.2094 | 0.3697 | 0.5200 | 0.4800 | 0.4800 | 1.0000 |
| hpm_lite | 30 | 0.2800 | 0.3547 | 0.2826 | 0.2700 | 0.5200 | 0.7300 | 1.0000 |
| hpm_lite | 100 | 0.4200 | 0.2302 | 0.5986 | 0.5600 | 0.5400 | 0.4400 | 1.0000 |
| hpm_lite | 300 | 0.2900 | 0.2094 | 0.3697 | 0.5300 | 0.4800 | 0.4700 | 1.0000 |
| hebbian | 30 | 0.2200 | 0.4692 | 0.0000 | 0.0000 | 0.5200 | 1.0000 | 1.0000 |
| hebbian | 100 | 0.4200 | 0.2152 | 0.5820 | 0.4800 | 0.5400 | 0.5200 | 1.0000 |
| hebbian | 300 | 0.3400 | 0.2683 | 0.4376 | 0.4900 | 0.4800 | 0.5100 | 1.0000 |

### Read/Use Verdict

- Structured readout fixes coexisting for epmem at 300 steps (0.000 CE exact vs 1.000 set exact), so the failure is decoder/composition, not memory retrieval.
- Conditional no-memory/local remains high at final budget: local 0.480, epmem no-retrieval 0.480. Fix the task before claiming qualifier reasoning.
- HPM-Lite vs epmem on coexisting final-budget CE exact delta is 0.0 points; recurrent component remains unjustified.

## Conditional Task Repair

This run repairs the conditional diagnostic and keeps the coexisting structured-readout baseline documented. It does not add a learned writer, larger model, JEPA, ANN, graph memory, GKA, Priming, RL, or extra HPM-Lite machinery.

Tasks: `coexisting,conditional_balanced,conditional_positive_only,conditional_contrastive`. Models: `local,epmem,hpm_lite,hebbian`. Write modes: `fact_token,oracle`. Controls: `normal,no_retrieval,shuffled_values,random_keys`.
Budgets: `30,100,300`. Seeds: `0,1,2,3,4`. Seq len/window: `512` / `64`. Top-k: `4`.
Device request: `cuda`. Platform: `Windows-10-10.0.19045-SP0`. Torch: `2.11.0+cu128`.
Leak checks passed on `80` generated examples.

Command:

- `python scripts/run_readuse_diagnostics.py --budgets 30,100,300 --seeds 0,1,2,3,4 --seq-len 512 --window 64 --batch-size 4 --eval-batches 5 --d-model 64 --layers 1 --heads 4 --device cuda`

Raw and summarized outputs: `runs/readuse_raw.csv`, `runs/readuse_summary.csv`.

### Coexisting Structured Baseline

Coexisting is readout-solved under structured set decoding when retrieval succeeds; this baseline is retained as a diagnostic control.

| model | budget | CE exact | retrieval top-k | structured set exact | per-value F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| local | 30 | 0.0000 |  | 0.0000 | 0.0000 |
| local | 100 | 0.0000 |  | 0.0000 | 0.0000 |
| local | 300 | 0.0000 |  | 0.0000 | 0.0000 |
| epmem | 30 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| epmem | 100 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| epmem | 300 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hpm_lite | 30 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hpm_lite | 100 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hpm_lite | 300 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hebbian | 30 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hebbian | 100 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| hebbian | 300 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

### Conditional Variant Metrics

| task | model | budget | exact | positive exact | negative exact | binding exact | no-value pred | value pred | retrieval top-k | use if retrieved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conditional_balanced | local | 30 | 0.4000 | 0.0000 | 0.8000 | 0.0000 | 0.8000 | 0.2000 |  |  |
| conditional_balanced | local | 100 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |  |  |
| conditional_balanced | local | 300 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |  |  |
| conditional_balanced | epmem | 30 | 0.3400 | 0.2400 | 0.4400 | 0.2400 | 0.4600 | 0.5400 | 1.0000 | 0.2400 |
| conditional_balanced | epmem | 100 | 0.3800 | 0.2600 | 0.5000 | 0.2600 | 0.5300 | 0.4700 | 1.0000 | 0.2600 |
| conditional_balanced | epmem | 300 | 0.4100 | 0.3000 | 0.5200 | 0.3000 | 0.5100 | 0.4900 | 1.0000 | 0.3000 |
| conditional_balanced | hpm_lite | 30 | 0.3300 | 0.2200 | 0.4400 | 0.2200 | 0.4700 | 0.5300 | 1.0000 | 0.2200 |
| conditional_balanced | hpm_lite | 100 | 0.3800 | 0.2600 | 0.5000 | 0.2600 | 0.5300 | 0.4700 | 1.0000 | 0.2600 |
| conditional_balanced | hpm_lite | 300 | 0.4300 | 0.3000 | 0.5600 | 0.3000 | 0.5400 | 0.4600 | 1.0000 | 0.3000 |
| conditional_balanced | hebbian | 30 | 0.1900 | 0.3800 | 0.0000 | 0.3800 | 0.0000 | 1.0000 | 1.0000 | 0.3800 |
| conditional_balanced | hebbian | 100 | 0.3500 | 0.2400 | 0.4600 | 0.2400 | 0.4900 | 0.5100 | 1.0000 | 0.2400 |
| conditional_balanced | hebbian | 300 | 0.4000 | 0.2800 | 0.5200 | 0.2800 | 0.5200 | 0.4800 | 1.0000 | 0.2800 |
| conditional_positive_only | local | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_positive_only | local | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_positive_only | local | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_positive_only | epmem | 30 | 0.5500 | 0.5500 | 0.0000 | 0.5500 | 0.0000 | 1.0000 | 1.0000 | 0.5500 |
| conditional_positive_only | epmem | 100 | 0.4500 | 0.4500 | 0.0000 | 0.4500 | 0.0000 | 1.0000 | 1.0000 | 0.4500 |
| conditional_positive_only | epmem | 300 | 0.5200 | 0.5200 | 0.0000 | 0.5200 | 0.0000 | 1.0000 | 1.0000 | 0.5200 |
| conditional_positive_only | hpm_lite | 30 | 0.5400 | 0.5400 | 0.0000 | 0.5400 | 0.0000 | 1.0000 | 1.0000 | 0.5400 |
| conditional_positive_only | hpm_lite | 100 | 0.4800 | 0.4800 | 0.0000 | 0.4800 | 0.0000 | 1.0000 | 1.0000 | 0.4800 |
| conditional_positive_only | hpm_lite | 300 | 0.5300 | 0.5300 | 0.0000 | 0.5300 | 0.0000 | 1.0000 | 1.0000 | 0.5300 |
| conditional_positive_only | hebbian | 30 | 0.6400 | 0.6400 | 0.0000 | 0.6400 | 0.0000 | 1.0000 | 1.0000 | 0.6400 |
| conditional_positive_only | hebbian | 100 | 0.5500 | 0.5500 | 0.0000 | 0.5500 | 0.0000 | 1.0000 | 1.0000 | 0.5500 |
| conditional_positive_only | hebbian | 300 | 0.5600 | 0.5600 | 0.0000 | 0.5600 | 0.0000 | 1.0000 | 1.0000 | 0.5600 |
| conditional_contrastive | local | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_contrastive | local | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_contrastive | local | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_contrastive | epmem | 30 | 0.5500 | 0.5500 | 0.0000 | 0.5500 | 0.0000 | 1.0000 | 1.0000 | 0.5500 |
| conditional_contrastive | epmem | 100 | 0.4500 | 0.4500 | 0.0000 | 0.4500 | 0.0000 | 1.0000 | 1.0000 | 0.4500 |
| conditional_contrastive | epmem | 300 | 0.5200 | 0.5200 | 0.0000 | 0.5200 | 0.0000 | 1.0000 | 1.0000 | 0.5200 |
| conditional_contrastive | hpm_lite | 30 | 0.5400 | 0.5400 | 0.0000 | 0.5400 | 0.0000 | 1.0000 | 1.0000 | 0.5400 |
| conditional_contrastive | hpm_lite | 100 | 0.4800 | 0.4800 | 0.0000 | 0.4800 | 0.0000 | 1.0000 | 1.0000 | 0.4800 |
| conditional_contrastive | hpm_lite | 300 | 0.5300 | 0.5300 | 0.0000 | 0.5300 | 0.0000 | 1.0000 | 1.0000 | 0.5300 |
| conditional_contrastive | hebbian | 30 | 0.6400 | 0.6400 | 0.0000 | 0.6400 | 0.0000 | 1.0000 | 1.0000 | 0.6400 |
| conditional_contrastive | hebbian | 100 | 0.5500 | 0.5500 | 0.0000 | 0.5500 | 0.0000 | 1.0000 | 1.0000 | 0.5500 |
| conditional_contrastive | hebbian | 300 | 0.5600 | 0.5600 | 0.0000 | 0.5600 | 0.0000 | 1.0000 | 1.0000 | 0.5600 |

### Conditional Repair Verdict

- Structured readout fixes coexisting for epmem at 300 steps (0.000 CE exact vs 1.000 set exact), so the failure is decoder/composition, not memory retrieval.
- Positive-only conditional: local 0.000, no-retrieval 0.000, epmem normal 0.520.
- Contrastive conditional requires memory in this run: local 0.000, no-retrieval 0.000, epmem 0.520.
- The oracle spot check matched fact-token at the final budget on the repaired conditional variants, so the parser writer is not the limiting factor here.
- Shuffled-values and random-keys both hurt repaired conditional exact accuracy. For epmem at 300 steps on contrastive, normal is 0.520, shuffled-values is 0.190, and random-keys is 0.250. Top-k is saturated because this run used top-k 4 with 4 slots, so top-1 and exact/control deltas are more informative than top-k here.
- HPM-Lite vs epmem: strongest final-budget repaired-conditional delta is 2.0 points on `conditional_balanced`; recurrence remains unjustified if this is below 5-10 points.
- Hebbian contrastive condition-binding exact at final budget is 0.560; low performance means simple association is not enough for qualifier binding.

## Condition Binding Structured Readout

This run tests whether a symbolic condition-binding operator can use the stored key-condition-value slots even when next-token CE decoding stays weak. It does not add a learned writer, larger model, JEPA, ANN, graph memory, GKA, Priming, RL, or extra HPM-Lite machinery.

Tasks: `conditional_positive_only,conditional_contrastive`. Models: `local,epmem,hpm_lite,hebbian`. Write modes: `fact_token,oracle`. Controls: `normal,no_retrieval,shuffled_values,random_keys`.
Budgets: `30,100,300`. Seeds: `0,1,2,3,4`. Seq len/window: `512` / `64`. Top-k: `4`.
Device request: `cuda`. Platform: `Windows-10-10.0.19045-SP0`. Torch: `2.11.0+cu128`.
Leak checks passed on `40` generated examples.

Commands run:

- `.\.venv-cuda\Scripts\python.exe -m py_compile .\hpm_lite\structured_readout.py .\hpm_lite\evaluate.py .\scripts\run_readuse_diagnostics.py`
- `.\.venv-cuda\Scripts\python.exe -m pytest -q`
- `.\.venv-cuda\Scripts\python.exe .\scripts\run_readuse_diagnostics.py --tasks conditional_positive_only,conditional_contrastive --write-modes fact_token,oracle --budgets 30,100,300 --seeds 0,1,2,3,4 --seq-len 512 --window 64 --batch-size 4 --eval-batches 5 --d-model 64 --layers 1 --heads 4 --device cuda`
- `.\.venv-cuda\Scripts\python.exe -m pytest -q`

Tests passed: final run `19 passed in 2.15s`. No tests failed.
Raw and summarized outputs: `runs/condition_binding_raw.csv` (`960` rows), `runs/condition_binding_summary.csv` (`192` rows).

### Coexisting Structured Baseline

Coexisting is readout-solved under structured set decoding when retrieval succeeds; this baseline is retained as a diagnostic control.

| model | budget | CE exact | retrieval top-k | structured set exact | per-value F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| local | 30 |  |  |  |  |
| local | 100 |  |  |  |  |
| local | 300 |  |  |  |  |
| epmem | 30 |  |  |  |  |
| epmem | 100 |  |  |  |  |
| epmem | 300 |  |  |  |  |
| hpm_lite | 30 |  |  |  |  |
| hpm_lite | 100 |  |  |  |  |
| hpm_lite | 300 |  |  |  |  |
| hebbian | 30 |  |  |  |  |
| hebbian | 100 |  |  |  |  |
| hebbian | 300 |  |  |  |  |

### Conditional Variant CE Metrics

| task | model | budget | exact | positive exact | negative exact | binding exact | no-value pred | value pred | retrieval top-k | use if retrieved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conditional_positive_only | local | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_positive_only | local | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_positive_only | local | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_positive_only | epmem | 30 | 0.5500 | 0.5500 | 0.0000 | 0.5500 | 0.0000 | 1.0000 | 1.0000 | 0.5500 |
| conditional_positive_only | epmem | 100 | 0.4500 | 0.4500 | 0.0000 | 0.4500 | 0.0000 | 1.0000 | 1.0000 | 0.4500 |
| conditional_positive_only | epmem | 300 | 0.5200 | 0.5200 | 0.0000 | 0.5200 | 0.0000 | 1.0000 | 1.0000 | 0.5200 |
| conditional_positive_only | hpm_lite | 30 | 0.5400 | 0.5400 | 0.0000 | 0.5400 | 0.0000 | 1.0000 | 1.0000 | 0.5400 |
| conditional_positive_only | hpm_lite | 100 | 0.4800 | 0.4800 | 0.0000 | 0.4800 | 0.0000 | 1.0000 | 1.0000 | 0.4800 |
| conditional_positive_only | hpm_lite | 300 | 0.5300 | 0.5300 | 0.0000 | 0.5300 | 0.0000 | 1.0000 | 1.0000 | 0.5300 |
| conditional_positive_only | hebbian | 30 | 0.6400 | 0.6400 | 0.0000 | 0.6400 | 0.0000 | 1.0000 | 1.0000 | 0.6400 |
| conditional_positive_only | hebbian | 100 | 0.5500 | 0.5500 | 0.0000 | 0.5500 | 0.0000 | 1.0000 | 1.0000 | 0.5500 |
| conditional_positive_only | hebbian | 300 | 0.5600 | 0.5600 | 0.0000 | 0.5600 | 0.0000 | 1.0000 | 1.0000 | 0.5600 |
| conditional_contrastive | local | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_contrastive | local | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_contrastive | local | 300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |  |  |
| conditional_contrastive | epmem | 30 | 0.5500 | 0.5500 | 0.0000 | 0.5500 | 0.0000 | 1.0000 | 1.0000 | 0.5500 |
| conditional_contrastive | epmem | 100 | 0.4500 | 0.4500 | 0.0000 | 0.4500 | 0.0000 | 1.0000 | 1.0000 | 0.4500 |
| conditional_contrastive | epmem | 300 | 0.5200 | 0.5200 | 0.0000 | 0.5200 | 0.0000 | 1.0000 | 1.0000 | 0.5200 |
| conditional_contrastive | hpm_lite | 30 | 0.5400 | 0.5400 | 0.0000 | 0.5400 | 0.0000 | 1.0000 | 1.0000 | 0.5400 |
| conditional_contrastive | hpm_lite | 100 | 0.4800 | 0.4800 | 0.0000 | 0.4800 | 0.0000 | 1.0000 | 1.0000 | 0.4800 |
| conditional_contrastive | hpm_lite | 300 | 0.5300 | 0.5300 | 0.0000 | 0.5300 | 0.0000 | 1.0000 | 1.0000 | 0.5300 |
| conditional_contrastive | hebbian | 30 | 0.6400 | 0.6400 | 0.0000 | 0.6400 | 0.0000 | 1.0000 | 1.0000 | 0.6400 |
| conditional_contrastive | hebbian | 100 | 0.5500 | 0.5500 | 0.0000 | 0.5500 | 0.0000 | 1.0000 | 1.0000 | 0.5500 |
| conditional_contrastive | hebbian | 300 | 0.5600 | 0.5600 | 0.0000 | 0.5600 | 0.0000 | 1.0000 | 1.0000 | 0.5600 |

### Conditional Repair Verdict

- Positive-only conditional: local 0.000, no-retrieval 0.000, epmem normal 0.520.
- Contrastive conditional requires memory in this run: local 0.000, no-retrieval 0.000, epmem 0.520.
- HPM-Lite vs epmem: strongest final-budget repaired-conditional delta is 1.0 points on `conditional_positive_only`; recurrence remains unjustified if this is below 5-10 points.
- Hebbian contrastive condition-binding exact at final budget is 0.560; low performance means simple association is not enough for qualifier binding.

### Symbolic Binding Final-Budget Controls

Local rows here are `local + external symbolic readout`, not pure local model behavior. `no_retrieval` disables symbolic memory readout and is reported as unavailable.

| task | write | model | control | CE exact | symbolic exact | symbolic_binding_hit_1_rate | exact available | ambiguous exact | slot acc | value acc | CE-symbolic gap | retrieval-symbolic gap |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conditional_positive_only | fact_token | local | normal | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |  |
| conditional_positive_only | fact_token | local | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_positive_only | fact_token | local | shuffled_values | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |  |
| conditional_positive_only | fact_token | local | random_keys | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |  |
| conditional_positive_only | fact_token | epmem | normal | 0.5200 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4800 | 0.4800 |
| conditional_positive_only | fact_token | epmem | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_positive_only | fact_token | epmem | shuffled_values | 0.1900 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.1900 | -0.1900 |
| conditional_positive_only | fact_token | epmem | random_keys | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.2500 | -0.2500 |
| conditional_positive_only | fact_token | hpm_lite | normal | 0.5300 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4700 | 0.4700 |
| conditional_positive_only | fact_token | hpm_lite | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_positive_only | fact_token | hpm_lite | shuffled_values | 0.1900 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.1900 | -0.1900 |
| conditional_positive_only | fact_token | hpm_lite | random_keys | 0.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.2600 | -0.2600 |
| conditional_positive_only | fact_token | hebbian | normal | 0.5600 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4400 | 0.4400 |
| conditional_positive_only | fact_token | hebbian | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_positive_only | fact_token | hebbian | shuffled_values | 0.2400 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.2400 | -0.2400 |
| conditional_positive_only | fact_token | hebbian | random_keys | 0.1300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.1300 | -0.1300 |
| conditional_positive_only | oracle | local | normal | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |  |
| conditional_positive_only | oracle | local | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_positive_only | oracle | local | shuffled_values | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |  |
| conditional_positive_only | oracle | local | random_keys | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |  |
| conditional_positive_only | oracle | epmem | normal | 0.5200 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4800 | 0.4800 |
| conditional_positive_only | oracle | epmem | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_positive_only | oracle | epmem | shuffled_values | 0.1900 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.1900 | -0.1900 |
| conditional_positive_only | oracle | epmem | random_keys | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.2500 | -0.2500 |
| conditional_positive_only | oracle | hpm_lite | normal | 0.5300 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4700 | 0.4700 |
| conditional_positive_only | oracle | hpm_lite | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_positive_only | oracle | hpm_lite | shuffled_values | 0.1900 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.1900 | -0.1900 |
| conditional_positive_only | oracle | hpm_lite | random_keys | 0.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.2600 | -0.2600 |
| conditional_positive_only | oracle | hebbian | normal | 0.5600 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4400 | 0.4400 |
| conditional_positive_only | oracle | hebbian | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_positive_only | oracle | hebbian | shuffled_values | 0.2400 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.2400 | -0.2400 |
| conditional_positive_only | oracle | hebbian | random_keys | 0.1300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.1300 | -0.1300 |
| conditional_contrastive | fact_token | local | normal | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |  |
| conditional_contrastive | fact_token | local | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_contrastive | fact_token | local | shuffled_values | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |  |
| conditional_contrastive | fact_token | local | random_keys | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |  |
| conditional_contrastive | fact_token | epmem | normal | 0.5200 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4800 | 0.4800 |
| conditional_contrastive | fact_token | epmem | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_contrastive | fact_token | epmem | shuffled_values | 0.1900 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.1900 | -0.1900 |
| conditional_contrastive | fact_token | epmem | random_keys | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.2500 | -0.2500 |
| conditional_contrastive | fact_token | hpm_lite | normal | 0.5300 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4700 | 0.4700 |
| conditional_contrastive | fact_token | hpm_lite | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_contrastive | fact_token | hpm_lite | shuffled_values | 0.1900 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.1900 | -0.1900 |
| conditional_contrastive | fact_token | hpm_lite | random_keys | 0.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.2600 | -0.2600 |
| conditional_contrastive | fact_token | hebbian | normal | 0.5600 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4400 | 0.4400 |
| conditional_contrastive | fact_token | hebbian | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_contrastive | fact_token | hebbian | shuffled_values | 0.2400 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.2400 | -0.2400 |
| conditional_contrastive | fact_token | hebbian | random_keys | 0.1300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.1300 | -0.1300 |
| conditional_contrastive | oracle | local | normal | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |  |
| conditional_contrastive | oracle | local | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_contrastive | oracle | local | shuffled_values | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |  |
| conditional_contrastive | oracle | local | random_keys | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |  |
| conditional_contrastive | oracle | epmem | normal | 0.5200 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4800 | 0.4800 |
| conditional_contrastive | oracle | epmem | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_contrastive | oracle | epmem | shuffled_values | 0.1900 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.1900 | -0.1900 |
| conditional_contrastive | oracle | epmem | random_keys | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.2500 | -0.2500 |
| conditional_contrastive | oracle | hpm_lite | normal | 0.5300 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4700 | 0.4700 |
| conditional_contrastive | oracle | hpm_lite | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_contrastive | oracle | hpm_lite | shuffled_values | 0.1900 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.1900 | -0.1900 |
| conditional_contrastive | oracle | hpm_lite | random_keys | 0.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.2600 | -0.2600 |
| conditional_contrastive | oracle | hebbian | normal | 0.5600 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4400 | 0.4400 |
| conditional_contrastive | oracle | hebbian | no_retrieval | 0.0000 |  |  |  |  |  |  |  |  |
| conditional_contrastive | oracle | hebbian | shuffled_values | 0.2400 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | -0.2400 | -0.2400 |
| conditional_contrastive | oracle | hebbian | random_keys | 0.1300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.1300 | -0.1300 |

### Symbolic Binding Verdict

Did symbolic condition binding hit 1.0?
Yes. Symbolic condition binding is essentially solved for epmem.
At budget 300 on conditional_contrastive with fact_token: epmem CE exact = 0.520; epmem symbolic condition exact = 1.000; epmem symbolic_binding_hit_1_rate = 1.000; exact_match_available_rate = 1.000.
hpm_lite CE exact = 0.530; hpm_lite symbolic condition exact = 1.000.
hebbian CE exact = 0.560; hebbian symbolic condition exact = 1.000.
local sanity: CE exact = 0.000; local + external symbolic readout exact = 1.000.
Symbolic condition binding is essentially solved. Memory contains the correct information. CE decoding/read-use is the bottleneck.
This is strong evidence that generic next-token CE decoding is the wrong readout for condition-binding memory.
If symbolic binding hits 1.0 while CE remains low, this is a write-up-worthy result.

## Learned Structured Readers v1

This run trains only tiny learned readers over typed memory slots. It does not train the local Transformer, CE decoder, memory writer, HPM-Lite recurrence, JEPA, ANN, graph memory, Priming, GKA, or RL.

Tasks: `conditional_positive_only,conditional_contrastive,coexisting`. Write modes: `fact_token,oracle`. Controls: `normal,no_retrieval,shuffled_values,random_keys`.
Budgets: `0,1,3,10,30,100`. Seeds: `0,1,2,3,4`. Seq len/window: `512` / `64`.
Reader: dim `64`, hidden `128`, layers `2`, dropout `0.0`, lr `0.001`.
Device request: `cuda`. Platform: `Windows-10-10.0.19045-SP0`. Torch: `2.11.0+cu128`.

Commands run:

- `.\.venv-cuda\Scripts\python.exe -m py_compile .\hpm_lite\structured_readout.py .\hpm_lite\evaluate.py .\scripts\run_structured_readers.py`
- `.\.venv-cuda\Scripts\python.exe -m pytest -q`
- `.\.venv-cuda\Scripts\python.exe .\scripts\run_structured_readers.py --tasks conditional_positive_only,conditional_contrastive,coexisting --write-modes fact_token,oracle --budgets 0,1,3,10,30,100 --seeds 0,1,2,3,4 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --reader-dim 64 --reader-hidden 128 --device cuda`
- `.\.venv-cuda\Scripts\python.exe -m pytest -q`

Tests passed: final run `21 passed in 2.26s`. No tests failed.
CE reference values are loaded from earlier CE diagnostics in `runs/condition_binding_summary.csv` and `runs/memfail_budget_summary.csv`; this reader-only run did not retrain the Transformer or CE decoder.

Raw and summarized outputs: `runs/structured_readers_raw.csv` (`720` rows), `runs/structured_readers_summary.csv` (`144` rows).

### Final-Budget Normal Results

| task | write | symbolic exact | learned exact | CE reference | learned-symbolic gap | learned-CE gap | params | steps>=0.90 | steps>=0.99 | train time sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conditional_positive_only | fact_token | 1.0000 | 1.0000 | 0.5600 | 0.0000 | 0.4400 | 102785 | 1.0000 | 2.6000 | 17.8784 |
| conditional_positive_only | oracle | 1.0000 | 1.0000 | 0.5600 | 0.0000 | 0.4400 | 102785 | 1.0000 | 2.6000 | 2.4379 |
| conditional_contrastive | fact_token | 1.0000 | 1.0000 | 0.5600 | 0.0000 | 0.4400 | 102785 | 1.0000 | 2.6000 | 17.8506 |
| conditional_contrastive | oracle | 1.0000 | 1.0000 | 0.5600 | 0.0000 | 0.4400 | 102785 | 1.0000 | 2.6000 | 2.7533 |
| coexisting | fact_token | 1.0000 | 1.0000 | 0.0100 | 0.0000 | 0.9900 | 70017 | 7.2000 | 26.0000 | 17.8381 |
| coexisting | oracle | 1.0000 | 1.0000 | 0.0100 | 0.0000 | 0.9900 | 70017 | 7.2000 | 26.0000 | 2.5704 |

### Final-Budget Controls

| task | write | control | symbolic exact | learned exact | CE reference | learned-CE gap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| conditional_positive_only | fact_token | normal | 1.0000 | 1.0000 | 0.5600 | 0.4400 |
| conditional_positive_only | fact_token | no_retrieval |  |  | 0.0000 |  |
| conditional_positive_only | fact_token | shuffled_values | 0.0000 | 0.0000 | 0.2600 | -0.2600 |
| conditional_positive_only | fact_token | random_keys | 0.0000 | 0.0000 | 0.2700 | -0.2700 |
| conditional_positive_only | oracle | normal | 1.0000 | 1.0000 | 0.5600 | 0.4400 |
| conditional_positive_only | oracle | no_retrieval |  |  | 0.0000 |  |
| conditional_positive_only | oracle | shuffled_values | 0.0000 | 0.0000 | 0.2600 | -0.2600 |
| conditional_positive_only | oracle | random_keys | 0.0000 | 0.0000 | 0.2700 | -0.2700 |
| conditional_contrastive | fact_token | normal | 1.0000 | 1.0000 | 0.5600 | 0.4400 |
| conditional_contrastive | fact_token | no_retrieval |  |  | 0.0000 |  |
| conditional_contrastive | fact_token | shuffled_values | 0.0000 | 0.0000 | 0.2600 | -0.2600 |
| conditional_contrastive | fact_token | random_keys | 0.0000 | 0.0000 | 0.2700 | -0.2700 |
| conditional_contrastive | oracle | normal | 1.0000 | 1.0000 | 0.5600 | 0.4400 |
| conditional_contrastive | oracle | no_retrieval |  |  | 0.0000 |  |
| conditional_contrastive | oracle | shuffled_values | 0.0000 | 0.0000 | 0.2600 | -0.2600 |
| conditional_contrastive | oracle | random_keys | 0.0000 | 0.0000 | 0.2700 | -0.2700 |
| coexisting | fact_token | normal | 1.0000 | 1.0000 | 0.0100 | 0.9900 |
| coexisting | fact_token | no_retrieval |  |  | 0.0000 |  |
| coexisting | fact_token | shuffled_values | 0.0000 | 0.0000 | 0.0100 | -0.0100 |
| coexisting | fact_token | random_keys | 0.0000 | 0.0000 | 0.0100 | -0.0100 |
| coexisting | oracle | normal | 1.0000 | 1.0000 | 0.0100 | 0.9900 |
| coexisting | oracle | no_retrieval |  |  | 0.0000 |  |
| coexisting | oracle | shuffled_values | 0.0000 | 0.0000 | 0.0100 | -0.0100 |
| coexisting | oracle | random_keys | 0.0000 | 0.0000 | 0.0100 | -0.0100 |

### Verdict

Can a small learned structured reader recover what CE decoding fails to use?
Yes for condition binding: learned condition exact reaches 1.000 while the CE reference is 0.560. Condition binding is solved by the structured reader; CE decoding/read-use is the bottleneck.
Yes for coexisting set readout: learned set exact reaches 1.000 while the CE reference is 0.010. Sequence next-token decoding is the wrong output operator for this set-valued query.
Does this support the architecture shift from HPM-Lite recurrence toward typed memory + structured readouts?
Yes. In these diagnostics, typed memory plus task-appropriate learned readers solves cases where HPM-Lite recurrence did not separate from epmem.

## Structured Reader Stress Suite v1

This run stress-tests learned structured readers over typed slots. It still trains only reader modules and separate reader embeddings; it does not train a backbone, CE decoder, memory writer, HPM-Lite recurrence, JEPA, ANN, graph memory, Priming, GKA, or RL.

Tasks: `conditional_contrastive_stress,coexisting_stress`. Write modes: `fact_token`. Controls: `normal,shuffled_values,random_keys,corrupt_conditions,corrupt_values`.
Budgets: `0,10,100`. Seeds: `0,1`. Slot counts: `4,16`. Hard negatives: `0,8`. Similarity modes: `none,mixed`. Slot order: `random`.
Reader: dim `64`, hidden `128`, layers `2`, lr `0.001`. Device request: `cuda`. Torch: `2.11.0+cu128`.

Command:

- `.\.venv-cuda\Scripts\python.exe .\scripts\run_structured_readers.py --tasks conditional_contrastive_stress,coexisting_stress --write-modes fact_token --budgets 0,10,100 --seeds 0,1 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --reader-dim 64 --reader-hidden 128 --device cuda --slot-counts 4,16 --num-hard-negatives 0,8 --similarity-modes none,mixed --slot-order random`

Raw and summarized outputs: `runs/structured_reader_stress_raw.csv` (`480` rows), `runs/structured_reader_stress_summary.csv` (`240` rows).

Compile/tests:

- `.\.venv-cuda\Scripts\python.exe -m py_compile .\hpm_lite\structured_readout.py .\hpm_lite\data.py .\hpm_lite\write_modes.py .\scripts\run_structured_readers.py`
- `.\.venv-cuda\Scripts\python.exe -m pytest -q` passed before and after the run: `25 passed`.

Stage B full stress was not launched. Stage A took about 11 minutes for 480 raw rows; the requested full Stage B grid is 22,400 raw rows with a larger 300-step budget and was projected to take several hours, so it was treated as too large under the staged-plan guardrail.

### Stage A Grid

| task | slot_count | hard_negatives | similarity | symbolic exact | learned exact | learned-symbolic gap | top1 slot | topk slot | hard FP |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coexisting_stress | 4 | 0 | mixed | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| coexisting_stress | 4 | 0 | none | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| coexisting_stress | 4 | 8 | mixed | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| coexisting_stress | 4 | 8 | none | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| coexisting_stress | 16 | 0 | mixed | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| coexisting_stress | 16 | 0 | none | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| coexisting_stress | 16 | 8 | mixed | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| coexisting_stress | 16 | 8 | none | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| conditional_contrastive_stress | 4 | 0 | mixed | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| conditional_contrastive_stress | 4 | 0 | none | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| conditional_contrastive_stress | 4 | 8 | mixed | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| conditional_contrastive_stress | 4 | 8 | none | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| conditional_contrastive_stress | 16 | 0 | mixed | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| conditional_contrastive_stress | 16 | 0 | none | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| conditional_contrastive_stress | 16 | 8 | mixed | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| conditional_contrastive_stress | 16 | 8 | none | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |

### Stress Aggregates

| grouping | value | learned exact | learned-symbolic gap |
| --- | --- | ---: | ---: |
| conditional slot_count | 16 | 1.0000 | 0.0000 |
| conditional slot_count | 4 | 1.0000 | 0.0000 |
| set slot_count | 16 | 1.0000 | 0.0000 |
| set slot_count | 4 | 1.0000 | 0.0000 |
| conditional hard_negatives | 0 | 1.0000 | 0.0000 |
| conditional hard_negatives | 8 | 1.0000 | 0.0000 |
| set hard_negatives | 0 | 1.0000 | 0.0000 |
| set hard_negatives | 8 | 1.0000 | 0.0000 |
| conditional similarity_mode | mixed | 1.0000 | 0.0000 |
| conditional similarity_mode | none | 1.0000 | 0.0000 |
| set similarity_mode | mixed | 1.0000 | 0.0000 |
| set similarity_mode | none | 1.0000 | 0.0000 |

### Controls

| control | symbolic exact | learned exact | learned-symbolic gap |
| --- | ---: | ---: | ---: |
| normal | 1.0000 | 1.0000 | 0.0000 |
| shuffled_values | 0.0000 | 0.0000 | 0.0000 |
| random_keys | 0.0000 | 0.0000 | 0.0000 |
| corrupt_conditions | 0.5000 | 0.5000 | 0.0000 |
| corrupt_values | 0.0000 | 0.0000 | 0.0000 |

### Failure Analysis

Worst normal learned-symbolic gap at final budget: `coexisting_stress` with slot_count `4`, hard negatives `0`, similarity `mixed`. Symbolic exact = 1.000, learned exact = 1.000, gap = 0.000.
`corrupt_values` mean learned exact = 0.000; value corruption remains a strong negative control.
`corrupt_conditions` mean conditional learned exact = 0.000; condition corruption hits binding as expected.
`random_keys` mean learned exact = 0.000; key corruption hits matching as expected.

### Verdict

Are learned structured readers still solving the task under ambiguity?
Yes at this stress level: final normal learned exact averages 1.000 while symbolic exact averages 1.000.
Where do they fail first?
The first visible failure is the worst-gap setting above: `coexisting_stress`, slot_count `4`, hard negatives `0`, similarity `mixed`.
Is the next step learned writing, reader robustness, or real-data slot extraction?
Next step should be real-data slot extraction or learned writing, because the reader survives this synthetic ambiguity suite.

## Noisy Slot Extraction / Learned Writer v1

This run trains only a tiny typed slot extractor. It freezes a small structured reader after a short oracle-slot pretrain and does not train a backbone, CE decoder, HPM recurrence, JEPA, ANN, graph memory, Priming, GKA, or RL.

Tasks: `noisy_conditional,noisy_coexisting`. Writers: `oracle,fact_token,learned_typed_extractor`. Noise levels: `clean,medium`. Marker rates: `1.0,0.5,0.0`. Distractors: `0,8`. Slot counts: `4,16`.
Budgets: `0,10,100`. Seeds: `0,1`. Reader pretrain steps: `100`. Device request: `cuda`. Platform: `Windows-10-10.0.19045-SP0`. Torch: `2.11.0+cu128`.
Projected requested rows: `864`. Raw rows saved: `864`. Summary rows: `432`.

Command:

- `.\.venv-cuda\Scripts\python.exe .\scripts\run_noisy_slot_extraction.py --tasks noisy_conditional,noisy_coexisting --writers oracle,fact_token,learned_typed_extractor --budgets 0,10,100 --seeds 0,1 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --extractor-dim 64 --extractor-hidden 128 --device cuda --noise-levels clean,medium --marker-rates 1.0,0.5,0.0 --distractor-counts 0,8 --slot-counts 4,16`

Compile/tests:

- `.\.venv-cuda\Scripts\python.exe -m py_compile .\hpm_lite\data.py .\hpm_lite\write_modes.py .\hpm_lite\structured_readout.py .\hpm_lite\noisy_extraction.py .\scripts\run_noisy_slot_extraction.py`
- `.\.venv-cuda\Scripts\python.exe -m pytest -q` passed before the run: `28 passed`.

### Final Budget By Marker Rate

| writer | marker_rate | slot_f1 | learned reader exact | symbolic exact | all slots exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| oracle | 0.5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| oracle | 0.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fact_token | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fact_token | 0.5 | 0.6567 | 0.3844 | 0.3844 | 0.0273 |
| fact_token | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| learned_typed_extractor | 1.0 | 0.8138 | 0.7109 | 0.7320 | 0.5375 |
| learned_typed_extractor | 0.5 | 0.7564 | 0.6297 | 0.6570 | 0.4469 |
| learned_typed_extractor | 0.0 | 0.7678 | 0.6391 | 0.6609 | 0.4359 |

### Final Budget By Noise Level

| writer | noise | slot_f1 | learned reader exact | symbolic exact |
| --- | --- | ---: | ---: | ---: |
| oracle | clean | 1.0000 | 1.0000 | 1.0000 |
| oracle | medium | 1.0000 | 1.0000 | 1.0000 |
| fact_token | clean | 0.5540 | 0.4611 | 0.4611 |
| fact_token | medium | 0.5505 | 0.4618 | 0.4618 |
| learned_typed_extractor | clean | 0.8062 | 0.6885 | 0.7182 |
| learned_typed_extractor | medium | 0.7525 | 0.6312 | 0.6484 |

### Verdict

Can typed memory survive learned/noisy slot extraction?
Not yet cleanly: learned extractor slot F1 is 0.7793 and marker-rate-0 learned-reader exact is 0.6391.
Does learned extraction beat fact_token when markers disappear?
Yes: at marker-rate 0.0, learned extraction reaches 0.639 versus fact_token 0.000.
Is the next bottleneck writer, reader, or real-data conversion?
Writer/extractor is the bottleneck: oracle slots remain high while learned slots do not.

## Noisy Slot Extraction v2: Order-Invariant Writer

This run adds a DETR-style set extractor with slot objectness and Hungarian-style matching over typed key/value/condition pointers. It trains only writer/extractor modules; the structured readers remain frozen, and no backbone, CE decoder, HPM recurrence, JEPA, ANN, graph memory, Priming, GKA, or RL is trained.

Tasks: `noisy_conditional,noisy_coexisting`. Writers: `oracle,fact_token,learned_typed_extractor,learned_set_extractor_v2`. Noise levels: `clean,medium`. Marker rates: `1.0,0.0`. Distractors: `0,8`. Slot counts: `4,8`. Max slots: `8,16`.
Budgets: `0,10,100`. Seeds: `0,1`. Objectness threshold: `0.5`. Lambda objectness: `1.0`. Device request: `cuda`. Platform: `Windows-10-10.0.19045-SP0`. Torch: `2.11.0+cu128`.
Projected requested rows: `1536`. Raw rows saved: `1536`. Summary rows: `768`.

Command:

- `.\\.venv-cuda\\Scripts\\python.exe .\\scripts\\run_noisy_slot_extraction.py --tasks noisy_conditional,noisy_coexisting --writers oracle,fact_token,learned_typed_extractor,learned_set_extractor_v2 --budgets 0,10,100 --seeds 0,1 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --extractor-dim 64 --extractor-hidden 128 --device cuda --noise-levels clean,medium --marker-rates 1.0,0.0 --distractor-counts 0,8 --slot-counts 4,8 --max-slots 8,16 --objectness-threshold 0.5`

Verification:

- `.\\.venv-cuda\\Scripts\\python.exe -m py_compile .\\hpm_lite\\data.py .\\hpm_lite\\write_modes.py .\\hpm_lite\\structured_readout.py .\\hpm_lite\\noisy_extraction.py .\\scripts\\run_noisy_slot_extraction.py` passed.
- `.\\.venv-cuda\\Scripts\\python.exe -m pytest -q` passed before and after Stage A: 30 tests passed.

### Final Budget Writer Comparison

| writer | slot F1 | all-slots exact | learned-reader exact | symbolic exact | slot-count accuracy | predicted slots | true slots | duplicate rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_writer | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6.0000 | 6.0000 | 0.0000 |
| fact_token_writer | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 3.0000 | 6.0000 | 0.0000 |
| learned_typed_extractor_v1 | 0.8050 | 0.5184 | 0.6871 | 0.7039 | 1.0000 | 6.0000 | 6.0000 | 0.0503 |
| learned_set_extractor_v2 | 0.8319 | 0.5145 | 0.7562 | 0.7936 | 0.6734 | 6.5117 | 6.0000 | 0.0410 |

### Performance By Marker Rate

| writer | marker_rate | slot F1 | learned-reader exact | symbolic exact | slot-count accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle_writer | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| oracle_writer | 0.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fact_token_writer | 1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fact_token_writer | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| learned_typed_extractor_v1 | 1.0 | 0.8206 | 0.7125 | 0.7281 | 1.0000 |
| learned_typed_extractor_v1 | 0.0 | 0.7894 | 0.6617 | 0.6797 | 1.0000 |
| learned_set_extractor_v2 | 1.0 | 0.8555 | 0.7973 | 0.8320 | 0.6887 |
| learned_set_extractor_v2 | 0.0 | 0.8082 | 0.7152 | 0.7551 | 0.6582 |

### Performance By Noise Level

| writer | noise | slot F1 | learned-reader exact | symbolic exact | slot-count accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| oracle_writer | clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| oracle_writer | medium | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fact_token_writer | clean | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| fact_token_writer | medium | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| learned_typed_extractor_v1 | clean | 0.8254 | 0.7016 | 0.7172 | 1.0000 |
| learned_typed_extractor_v1 | medium | 0.7846 | 0.6727 | 0.6906 | 1.0000 |
| learned_set_extractor_v2 | clean | 0.8546 | 0.7812 | 0.8156 | 0.6914 |
| learned_set_extractor_v2 | medium | 0.8091 | 0.7312 | 0.7715 | 0.6555 |

### Performance By Slot Count

| writer | slot_count | max_slots | slot F1 | learned-reader exact | symbolic exact | slot-count accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_writer | 4 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| oracle_writer | 4 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| oracle_writer | 8 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| oracle_writer | 8 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fact_token_writer | 4 | 8 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| fact_token_writer | 4 | 16 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| fact_token_writer | 8 | 8 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| fact_token_writer | 8 | 16 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| learned_typed_extractor_v1 | 4 | 8 | 0.8314 | 0.7195 | 0.7328 | 1.0000 |
| learned_typed_extractor_v1 | 4 | 16 | 0.8314 | 0.7195 | 0.7328 | 1.0000 |
| learned_typed_extractor_v1 | 8 | 8 | 0.7786 | 0.6547 | 0.6750 | 1.0000 |
| learned_typed_extractor_v1 | 8 | 16 | 0.7786 | 0.6547 | 0.6750 | 1.0000 |
| learned_set_extractor_v2 | 4 | 8 | 0.8393 | 0.7758 | 0.8117 | 0.5625 |
| learned_set_extractor_v2 | 4 | 16 | 0.8355 | 0.7625 | 0.7953 | 0.6312 |
| learned_set_extractor_v2 | 8 | 8 | 0.8145 | 0.7234 | 0.7492 | 1.0000 |
| learned_set_extractor_v2 | 8 | 16 | 0.8381 | 0.7633 | 0.8180 | 0.5000 |

### Verdict

Can typed memory survive unordered learned slot extraction?
Not yet cleanly: v2 learned-reader exact at marker-rate 0.0 is 0.7152, with symbolic-on-v2-slots exact 0.7551 and slot F1 0.8319.
Does v2 beat fact_token when markers disappear?
Yes: at marker-rate 0.0, v2 reaches 0.715 versus fact_token 0.000.
Does v2 remove the known-slot-count / canonical-order cheat?
Partially. The v2 extractor predicts an unordered set and uses objectness at inference instead of the exact slot count. In this sweep each configuration is still trained/evaluated at a fixed true slot-count setting, so variable-count generalization remains a limitation.
At marker-rate 0.0, v1 canonical exact is 0.662; v2 unordered exact is 0.715.
Is the next bottleneck slot count, slot field extraction, duplicate suppression, or real-data conversion?
Slot count/objectness is the leading bottleneck: v2 slot-count accuracy averages 0.673.

## Writer v2 Bottleneck Decomposition

This diagnostic keeps the v2 architecture fixed and evaluates alternate masks/fields to separate objectness, slot-count calibration, duplicate behavior, and pointer-field quality. It does not add a new writer, larger backbone, JEPA, ANN, graph memory, RL, or HPM recurrence.

Tasks: `noisy_conditional,noisy_coexisting`. Writers: `learned_set_extractor_v2`. Noise levels: `medium`. Marker rates: `0.0`. Distractors: `8`. Slot counts: `4,8`. Max slots: `8,16`.
Budgets: `100`. Seeds: `0,1`. Lambda objectness values: `0.25,0.5,1.0,2.0,5.0`. Eval modes: `normal_v2,oracle_count_topk,oracle_objectness,oracle_fields,oracle_count_and_fields`. Threshold sweep: `0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9`. Device request: `cuda`. Torch: `2.11.0+cu128`.
Projected requested rows: `1200`. Raw rows saved: `1200`. Summary rows: `600`.

Command:

- `.\.venv-cuda\Scripts\python.exe .\scripts\run_noisy_slot_extraction.py --tasks noisy_conditional,noisy_coexisting --writers learned_set_extractor_v2 --budgets 100 --seeds 0,1 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --extractor-dim 64 --extractor-hidden 128 --device cuda --noise-levels medium --marker-rates 0.0 --distractor-counts 8 --slot-counts 4,8 --max-slots 8,16 --objectness-threshold 0.5 --lambda-obj-values 0.25,0.5,1.0,2.0,5.0 --threshold-sweep 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --eval-modes normal_v2,oracle_count_topk,oracle_objectness,oracle_fields,oracle_count_and_fields`

Verification:

- `.\.venv-cuda\Scripts\python.exe -m py_compile .\hpm_lite\noisy_extraction.py .\scripts\run_noisy_slot_extraction.py` passed.
- `.\.venv-cuda\Scripts\python.exe -m pytest -q` passed before and after the diagnostic.

### Eval Mode Summary

| eval mode | learned exact | symbolic exact | slot F1 | all-slots exact | slot-count acc | objectness F1 | objectness margin | gain vs normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal_v2 | 0.3981 | 0.4288 | 0.5517 | 0.0181 | 0.4169 | 0.7768 | 0.5211 | 0.0000 |
| oracle_count_topk | 0.3634 | 0.3887 | 0.5377 | 0.0175 | 1.0000 | 0.7508 | 0.5211 | -0.0347 |
| oracle_objectness | 0.4966 | 0.5312 | 0.6348 | 0.1306 | 1.0000 | 1.0000 | 0.5211 | 0.0984 |
| oracle_fields | 0.7238 | 0.7622 | 0.8202 | 0.3031 | 0.4169 | 0.7768 | 0.5211 | 0.3256 |
| oracle_count_and_fields | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7508 | 0.5211 | 0.6019 |

### Lambda Objectness Sweep

| lambda_obj | normal exact | oracle_count_topk exact | oracle_objectness exact | oracle_fields exact | objectness F1 | slot-count acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.4078 | 0.3719 | 0.4781 | 0.7500 | 0.7851 | 0.2750 |
| 0.5 | 0.3953 | 0.3734 | 0.4859 | 0.7141 | 0.7756 | 0.4719 |
| 1 | 0.4203 | 0.3750 | 0.5156 | 0.7312 | 0.7782 | 0.4219 |
| 2 | 0.3875 | 0.3656 | 0.5187 | 0.6859 | 0.7641 | 0.5406 |
| 5 | 0.3797 | 0.3312 | 0.4844 | 0.7375 | 0.7813 | 0.3750 |

### Threshold Sweep

Best threshold: `0.0500`. Best threshold exact: `0.4447`. Best threshold slot F1: `0.5667`.

### Verdict

normal_v2 exact = 0.3981
oracle_count_topk exact = 0.3634
oracle_objectness exact = 0.4966
oracle_fields exact = 0.7238
oracle_count_and_fields exact = 1.0000
objectness_margin = 0.5211
objectness_f1 = 0.7768
slot_count_accuracy = 0.4169
duplicate_slot_rate = 0.0732

Is the v2 bottleneck objectness, field extraction, calibration, or duplicate suppression?
Field pointer extraction is the largest bottleneck: oracle_fields gains 0.326 over normal.
Should Writer v3 use AdaSlot-style adaptive selection, high-recall span candidates, or better pointer fields?
Prefer better pointer fields or high-recall span candidates first.

## Writer v3: Field-Candidate Proposer and Slot Assembler

Writer v3 keeps the model tiny and replaces direct all-token field pointing with high-recall field candidates followed by tuple assembly over candidate pools. It trains only writer modules and does not add a larger backbone, HPM recurrence, JEPA, ANN, graph memory, or RL.

Tasks: `noisy_conditional,noisy_coexisting`. Writers: `learned_set_extractor_v2,writer_v3_oracle_candidates,writer_v3_learned_candidates,writer_v3_oracle_candidates_plus_noise`. Noise levels: `medium`. Marker rates: `0.0`. Distractors: `8`. Slot counts: `4,8`. Max slots: `8,16`. Candidate K: `4,8,16,32`.
Budgets: `100`. Seeds: `0,1`. Candidate loss weight: `1.0`. Device request: `cuda`. Torch: `2.11.0+cu128`.
Projected requested rows: `256`. Raw rows saved: `256`. Summary rows: `128`.

Command:

- `.\.venv-cuda\Scripts\python.exe .\scripts\run_noisy_slot_extraction.py --tasks noisy_conditional,noisy_coexisting --writers learned_set_extractor_v2,writer_v3_oracle_candidates,writer_v3_learned_candidates,writer_v3_oracle_candidates_plus_noise --budgets 100 --seeds 0,1 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --extractor-dim 64 --extractor-hidden 128 --device cuda --noise-levels medium --marker-rates 0.0 --distractor-counts 8 --slot-counts 4,8 --max-slots 8,16 --candidate-k-values 4,8,16,32 --candidate-loss-weight 1.0`

Verification:

- `.\.venv-cuda\Scripts\python.exe -m py_compile .\hpm_lite\noisy_extraction.py .\scripts\run_noisy_slot_extraction.py` passed.
- `.\.venv-cuda\Scripts\python.exe -m pytest -q` passed before and after the diagnostic.

### Writer Comparison

| writer | learned exact | symbolic exact | slot F1 | all-slots exact | candidate recall | candidate precision | gain over v2 | gap to oracle candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_set_extractor_v2 | 0.4207 | 0.4445 | 0.5563 | 0.0168 | 0.0000 | 0.0000 |  |  |
| writer_v3_oracle_candidates | 0.0254 | 0.0348 | 0.0972 | 0.0000 | 0.9375 | 0.6095 | -0.3953 | 0.0000 |
| writer_v3_learned_candidates | 0.0250 | 0.0348 | 0.1004 | 0.0000 | 0.9286 | 0.6443 | -0.3957 | 0.0004 |
| writer_v3_oracle_candidates_plus_noise | 0.0254 | 0.0348 | 0.0972 | 0.0000 | 0.9375 | 0.6095 | -0.3953 | 0.0000 |

### Candidate K Sweep

| candidate_k | v2 exact | v3 oracle-candidate exact | v3 learned-candidate exact | key recall | cond recall | value recall | all-field recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.4219 | 0.0469 | 0.0406 | 0.7223 | 0.8586 | 0.7500 | 0.7307 |
| 8 | 0.4203 | 0.0219 | 0.0234 | 0.9793 | 0.9822 | 1.0000 | 0.9837 |
| 16 | 0.4203 | 0.0172 | 0.0187 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 32 | 0.4203 | 0.0156 | 0.0172 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Verdict

v2 baseline exact = 0.4207
v3 oracle-candidate exact = 0.0254
v3 learned-candidate exact = 0.0250
v3 oracle-candidates-plus-noise exact = 0.0254
v3 gain over v2 = -0.3957
v3 learned gap to oracle candidates = 0.0004

Did field candidates fix the v2 pointer bottleneck?
No: even oracle candidates do not clearly improve, so tuple assembly is likely the bottleneck.
Is the next bottleneck candidate recall, tuple assembly, or objectness?
Tuple assembly is the next bottleneck.
Should the next paper-inspired step be HGERE-style high-recall pruning, SPN4RE-style tuple set prediction, MILIE-style iterative completion, or AdaSlot-style adaptive selection?
Use SPN4RE-style tuple set prediction improvements next.

## Tuple Assembler Debug

This diagnostic keeps the candidate setting tiny and checks whether v3 failed because of candidate-index construction, matching/objectness, independent field heads, or the tuple assembly objective. It does not add AdaSlot, MILIE, real datasets, HPM recurrence, JEPA, ANN, graph memory, or RL.

Tasks: `noisy_conditional`. Writers: `writer_v3_oracle_candidates_sanity,writer_v3_oracle_candidates,spn_tuple_assembler_oracle_candidates`. Noise levels: `medium`. Marker rates: `0.0`. Distractors: `8`. Slot counts: `4`. Max slots: `4`. Candidate K: `4,8`.
Budgets: `0,10,100,300`. Seeds: `0,1`. Debug examples: `10`. Device request: `cuda`. Torch: `2.11.0+cu128`.
Projected requested rows: `160`. Raw rows saved: `160`. Summary rows: `80`.

Command:

- `.\.venv-cuda\Scripts\python.exe .\scripts\run_noisy_slot_extraction.py --tasks noisy_conditional --writers writer_v3_oracle_candidates_sanity,writer_v3_oracle_candidates,spn_tuple_assembler_oracle_candidates --budgets 0,10,100,300 --seeds 0,1 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --extractor-dim 64 --extractor-hidden 128 --device cuda --noise-levels medium --marker-rates 0.0 --distractor-counts 8 --slot-counts 4 --max-slots 4 --candidate-k-values 4,8 --tuple-debug-examples 10`

Verification:

- `.\.venv-cuda\Scripts\python.exe -m py_compile .\hpm_lite\noisy_extraction.py .\scripts\run_noisy_slot_extraction.py` passed.
- `.\.venv-cuda\Scripts\python.exe -m pytest -q` passed before and after the diagnostic.

### Sanity Modes

| eval mode | exact | slot F1 | key cand acc | cond cand acc | value cand acc | tuple acc | matched tuple acc | mean gold cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent_field_heads_current | 0.0563 | 0.0444 | 0.2313 | 0.2797 | 0.2781 | 0.0141 | 0.0000 | 3.8413 |
| no_objectness_true_count | 0.0563 | 0.0444 | 0.2313 | 0.2797 | 0.2781 | 0.0141 | 0.0000 | 3.8413 |
| no_hungarian_canonical_debug | 0.0563 | 0.0444 | 0.2313 | 0.2797 | 0.2781 | 0.0141 | 0.0000 | 3.8413 |
| gold_key_only | 0.0938 | 0.0625 | 0.2313 | 0.2797 | 0.2781 | 0.0141 | 0.0000 | 3.8413 |
| gold_cond_only | 0.0938 | 0.0656 | 0.2313 | 0.2797 | 0.2781 | 0.0141 | 0.0000 | 3.8413 |
| gold_value_only | 0.0500 | 0.0594 | 0.2313 | 0.2797 | 0.2781 | 0.0141 | 0.0000 | 3.8413 |
| gold_key_cond | 0.3000 | 0.2781 | 0.2313 | 0.2797 | 0.2781 | 0.0141 | 0.0000 | 3.8413 |
| gold_all_fields | 1.0000 | 1.0000 | 0.2313 | 0.2797 | 0.2781 | 0.0141 | 0.0000 | 3.8413 |

### SPN Tuple Scorer

| writer | exact | slot F1 | tuple exact | tuple accuracy | gain over independent |
| --- | ---: | ---: | ---: | ---: | ---: |
| spn_tuple_assembler_oracle_candidates | 0.2937 | 0.3812 | 0.2937 | 0.3812 | 0.2375 |

### Verdict

oracle-candidate sanity no-objectness exact = 0.0563
canonical debug exact = 0.0563
gold_key_cond exact = 0.3000
gold_all_fields exact = 1.0000
SPN tuple exact = 0.2937

Was v3 failure an implementation/matching bug or a real tuple-assembly objective failure?
The assembler objective/factorization is failing under oracle candidates; gold field plumbing is sane.
Do independent field heads work under oracle candidates?
No.
Does SPN-style whole-tuple scoring fix assembly?
No.
What should Writer v4 be?
Writer v4 should move toward SPN-style tuple scoring.

## Writer v4 Contextual Tuple Edge Scorer

Writer v4 keeps the typed candidate setup but replaces isolated tuple embeddings with contextual relation evidence: token context, absolute and relative positions, field order bits, pooled text between fields, and local windows around each candidate field. This run trains only the tuple scorer/writer modules and does not add AdaSlot, MILIE, real datasets, HPM recurrence, JEPA, ANN, graph memory, RL, or a larger backbone.

Tasks: `noisy_conditional`. Writers: `contextual_tuple_oracle_candidates,contextual_tuple_oracle_candidates_plus_hard_negatives,contextual_tuple_gold_key_cond,contextual_tuple_gold_all_fields`. Noise levels: `medium`. Marker rates: `0.0`. Distractors: `8`. Slot counts: `4`. Max slots: `4`. Candidate K: `4,8`.
Budgets: `0,10,100,300`. Seeds: `0,1`. Candidate loss weight: `1.0`. Debug examples: `10`. Device request: `cuda`. Torch: `2.11.0+cu128`.
Projected requested rows: `64`. Raw rows saved: `64`. Summary rows: `32`.

Command:

- `.\.venv-cuda\Scripts\python.exe .\scripts\run_noisy_slot_extraction.py --tasks noisy_conditional --writers contextual_tuple_oracle_candidates,contextual_tuple_oracle_candidates_plus_hard_negatives,contextual_tuple_gold_key_cond,contextual_tuple_gold_all_fields --budgets 0,10,100,300 --seeds 0,1 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --extractor-dim 64 --extractor-hidden 128 --device cuda --noise-levels medium --marker-rates 0.0 --distractor-counts 8 --slot-counts 4 --max-slots 4 --candidate-k-values 4,8 --tuple-debug-examples 10`

Verification:

- `.\.venv-cuda\Scripts\python.exe -m py_compile .\hpm_lite\noisy_extraction.py .\scripts\run_noisy_slot_extraction.py` passed.
- `.\.venv-cuda\Scripts\python.exe -m pytest -q` passed before and after the diagnostic.

Previous tuple-assembler baselines from the prior debug run: independent field heads exact = `0.0563`; SPN tuple scorer exact = `0.2937`.

### Writer Comparison

| writer | exact | slot F1 | all-slots exact | tuple AUC | pos score | neg score | score margin | gain vs SPN | gain vs independent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| contextual_tuple_oracle_candidates | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 8.5708 | -10.7727 | 19.3435 | 0.7063 | 0.9437 |
| contextual_tuple_oracle_candidates_plus_hard_negatives | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 8.5708 | -11.1223 | 19.6931 | 0.7063 | 0.9437 |
| contextual_tuple_gold_key_cond | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 8.5708 | -10.7727 | 19.3435 | 0.7063 | 0.9437 |
| contextual_tuple_gold_all_fields | 1.0000 | 1.0000 | 1.0000 |  |  |  |  | 0.7063 | 0.9437 |

### Candidate K Sweep

| candidate_k | oracle candidates exact | oracle + hard negatives exact | gold key+cond exact | gold all fields exact | hard-neg FP rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0004 |
| 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0001 |

### Verdict

independent heads baseline exact = 0.0563
old SPN tuple baseline exact = 0.2937
contextual oracle-candidate exact = 1.0000
contextual oracle-candidates-plus-hard-negatives exact = 1.0000
contextual gold_key_cond exact = 1.0000
contextual gold_all_fields exact = 1.0000

Did contextual relation evidence fix tuple assembly?
Yes under clean oracle candidates: contextual relation evidence brings assembly near the symbolic/gold upper bound.
Can oracle candidates now be assembled correctly?
Yes.
Is the next bottleneck hard negative discrimination, learned candidates, or iterative completion?
Oracle assembly is healthy; learned candidate extraction is the next thing to re-enable.
Should we now move toward SPN4RE-style set prediction, MILIE-style iterative completion, or learned candidate extraction?
Move to learned candidate extraction next.

## Writer v4.5 Residual Failure Fix

This residual sweep keeps only `v4_condition_v2_full` and searches the remaining candidate/weight/augmentation settings. No new architecture or real-data move is introduced.

Task: `noisy_conditional`. Noise: `hard`. Marker rate: `0.0`. Distractors: `16`. Slot count: `8`. Max slots: `8`. Budget: `300`. Seeds: `0,1,2`.
Candidate K: `16,24,32`. Guideline weights: `0.5,1.0,2.0`. Simplified aux weights: `0.0,0.25,0.5`. Template augmentations: `heavy,extreme`. Template split: `heldout`.
Raw rows saved: `162`. Summary rows: `54`.

Command:

- `.\.venv-cuda\Scripts\python.exe .\scripts\run_noisy_slot_extraction.py --tasks noisy_conditional --writers v4_condition_v2_full --budgets 300 --seeds 0,1,2 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --extractor-dim 64 --extractor-hidden 128 --device cuda --noise-levels hard --marker-rates 0.0 --distractor-counts 16 --slot-counts 8 --max-slots 8 --candidate-k-values 16,24,32 --candidate-loss-weight 1.0 --tuple-loss-weight 1.0 --rank-loss-weight 0.5 --template-split heldout --template-augmentations heavy,extreme --simplified-aux-weight-values 0.0,0.25,0.5 --guideline-loss-weight-values 0.5,1.0,2.0 --tuple-debug-examples 20 --output-prefix writer_v45_residual_fix`

### Top Configurations

| rank | exact | condition recall | condition miss | value miss | tuple scoring error | candidate_k | guideline | simplified aux | augmentation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.9417 | 0.9865 | 0.0135 | 0.0000 | 0.4083 | 16 | 2.0 | 0.5 | extreme |
| 2 | 0.9333 | 0.9792 | 0.0208 | 0.0000 | 0.4750 | 16 | 0.5 | 0.5 | extreme |
| 3 | 0.9250 | 0.9823 | 0.0177 | 0.0000 | 0.4667 | 16 | 0.5 | 0.0 | extreme |
| 4 | 0.9250 | 0.9813 | 0.0187 | 0.0000 | 0.4083 | 16 | 2.0 | 0.0 | extreme |
| 5 | 0.9250 | 0.9740 | 0.0260 | 0.0000 | 0.4000 | 16 | 0.5 | 0.25 | extreme |
| 6 | 0.9167 | 0.9812 | 0.0187 | 0.0000 | 0.4917 | 16 | 2.0 | 0.25 | extreme |
| 7 | 0.9167 | 0.9708 | 0.0292 | 0.0000 | 0.3833 | 16 | 0.5 | 0.0 | heavy |
| 8 | 0.9083 | 0.9792 | 0.0208 | 0.0000 | 0.4333 | 16 | 1.0 | 0.5 | extreme |
| 9 | 0.9000 | 0.9771 | 0.0229 | 0.0000 | 0.4583 | 16 | 1.0 | 0.0 | extreme |
| 10 | 0.9000 | 0.9812 | 0.0187 | 0.0000 | 0.4833 | 16 | 1.0 | 0.25 | extreme |
| 11 | 0.8667 | 0.9698 | 0.0302 | 0.0000 | 0.4333 | 16 | 2.0 | 0.0 | heavy |
| 12 | 0.8500 | 0.9708 | 0.0292 | 0.0031 | 0.4917 | 16 | 0.5 | 0.25 | heavy |

### Best Setting

heldout exact = 0.9417
condition recall = 0.9865
value miss rate = 0.0000
tuple scoring error rate = 0.4083
candidate_k = 16
guideline_loss_weight = 2.0
simplified_aux_weight = 0.5
template_augmentation = extreme

### Verdict

Synthetic heldout is repaired under the requested rule.
Higher guideline weight appears useful: condition semantics/role supervision helped.
Extreme augmentation appears useful: template coverage helped.

## Writer v4.5 Freeze + Speed Optimization

This freezes the repaired v4.5 setting and compares full K^3 tuple scoring with optional pair-beam pruning. No new architecture is added.

Frozen defaults: candidate_k=16, guideline_loss_weight=2.0, simplified_aux_weight=0.5, template_augmentation=extreme. Raw rows: `6`. Summary rows: `2`.

Commands:

- `.\.venv-cuda\Scripts\python.exe .\scripts\run_noisy_slot_extraction.py --tasks noisy_conditional --writers v4_condition_v2_full --budgets 300 --seeds 0,1,2 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --extractor-dim 64 --extractor-hidden 128 --device cuda --noise-levels hard --marker-rates 0.0 --distractor-counts 16 --slot-counts 8 --max-slots 8 --candidate-k-values 16 --candidate-loss-weight 1.0 --tuple-loss-weight 1.0 --rank-loss-weight 0.5 --template-split heldout --template-augmentations extreme --simplified-aux-weight-values 0.5 --guideline-loss-weight-values 2.0 --tuple-debug-examples 20 --output-prefix writer_v45_frozen_baseline`
- `.\.venv-cuda\Scripts\python.exe .\scripts\run_noisy_slot_extraction.py --tasks noisy_conditional --writers v4_condition_v2_full --budgets 300 --seeds 0,1,2 --seq-len 512 --window 64 --batch-size 4 --eval-batches 10 --extractor-dim 64 --extractor-hidden 128 --device cuda --noise-levels hard --marker-rates 0.0 --distractor-counts 16 --slot-counts 8 --max-slots 8 --candidate-k-values 16 --candidate-loss-weight 1.0 --tuple-loss-weight 1.0 --rank-loss-weight 0.5 --template-split heldout --template-augmentations extreme --simplified-aux-weight-values 0.5 --guideline-loss-weight-values 2.0 --tuple-debug-examples 20 --output-prefix writer_v45_pairbeam_b8 --tuple-pruning pair_beam --pair-beam-size 8`

### Comparison

| tuple pruning | beam | exact | condition recall | tuple candidates | tuple scorer time | proposer time | reader time | examples/sec | GPU MB | tuple-time speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0 | 0.9083 | 0.9802 | 16298.6667 | 0.0223 | 0.0541 | 0.0028 | 22.9195 | 631.5444 | 1.0000 |
| pair_beam | 8 | 0.1250 | 0.9802 | 512.0000 | 0.1524 | 0.0352 | 0.0030 | 22.9482 | 631.5444 | 0.1461 |

### Verdict

Frozen baseline exact = 0.9083
Frozen baseline condition recall = 0.9802
pair_beam did not pass the accuracy gate in this run, so full K^3 should remain the default.
K=24/32 remain officially rejected for default runs; they require `--slow-sweep`.

## Integrated Memory Model v1

This compares a small local causal Transformer answer baseline against a typed-memory pipeline: Writer v4.5 with K=16, full K^3 tuple scoring, typed memory slots, and a learned structured reader. The writer/reader path is trained with extraction/readout objectives; the Transformer baseline is trained with answer CE.

Raw rows: `6`. Summary rows: `2`.

Command:

- `.\.venv-cuda\Scripts\python.exe .\scripts\run_integrated_memory.py --tasks noisy_conditional --models transformer_baseline,integrated_memory_v1_2_condition_v3_cond32 --writer-mode frozen --budgets 1000 --seeds 0,1,2 --seq-len 1024 --window 64 --batch-size 4 --eval-batches 10 --d-model 64 --layers 1 --heads 4 --device cuda --noise-levels hard --marker-rates 0.0 --distractor-counts 32 --slot-counts 16 --max-slots 16 --candidate-k 16 --template-split random`

### Final Budget Results

| task | split | model | exact | CE | writer exact | slot F1 | condition recall | train time | ex/sec | params | gain vs Transformer |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| noisy_conditional | random | integrated_memory_v1_2_condition_v3_cond32 | 0.9583 | 0.6779 | 0.5917 | 0.9667 | 0.9990 | 362.3622 | 11.0404 | 370003.0000 | 0.9417 |
| noisy_conditional | random | transformer_baseline | 0.0167 | 5.7179 |  |  |  | 14.5097 | 276.0157 | 146368.0000 |  |

### Verdict

Mean integrated-memory gain at budget 1000: 
Worst integrated-memory exact at budget 1000: 
Integrated memory loses to the Transformer baseline; stop scaling and inspect the integration.
All requested task/split cells are strong enough to consider a small scale-up check.
