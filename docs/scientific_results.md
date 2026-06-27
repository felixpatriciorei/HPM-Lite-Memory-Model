# Scientific Results

## Current result table

| Setting | Seq len | Local exact | HPM exact | Gain | Local CE | HPM CE |
|---|---:|---:|---:|---:|---:|---:|
| Oracle write | 512 | 0.0063 | 1.0000 | +0.9938 | 7.37 | 0.00 |
| Oracle write | 2048 | 0.0000 | 1.0000 | +1.0000 | 7.35 | 0.00 |
| Null-slot memory | 4096 | 0.0000 | 1.0000 | +1.0000 | 13.71 | 0.00 |
| Null-slot memory | 8192 | 0.0000 | 1.0000 | +1.0000 | 23.87 | 0.00 |
| Learned writer | 512 | 0.0125 | 1.0000 | +0.9875 | 6.27 | 0.00 |
| Learned writer | 2048 | 0.0000 | 1.0000 | +1.0000 | 6.63 | ~0.00 |

## Learned-writer stage

| Seq len | Writer recall | Missed fact rate | False write rate | Retrieval top1 |
|---:|---:|---:|---:|---:|
| 512 | 0.9922 | not recorded in summary table | not recorded in summary table | 1.0000 |
| 2048 | 0.99375 | 0.00625 | 0.00625 | 1.0000 |

## Interpretation

The local baseline fails because the relevant fact is outside the local window. HPM-Lite succeeds because episodic memory preserves and retrieves the key-value fact.

The learned-writer result is important because it reduces oracle spoon-feeding. The model learns which token positions should become memory slots from synthetic supervision.

## Caveats

The current evidence is still early: mostly single-seed, synthetic, and not fully autonomous. The next required evidence is ablation and multi-seed testing.
