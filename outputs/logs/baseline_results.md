# Sequence Modeling Results

## Setting
- Dataset: Amazon Electronics
- Implicit feedback: rating >= 3
- Max sequence length: 50
- Epochs: 3
- Eval users: 500
- Seen-item masking: enabled
- Metrics: HR@10, NDCG@10, HR@20, NDCG@20

## Validation Results
| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| GRU4Rec | 0.0980 | 0.0603 | 0.1680 | 0.0781 |
| SASRec | 0.0200 | 0.0082 | 0.0460 | 0.0147 |
| Time-Aware SASRec | 0.1320 | 0.0702 | 0.2040 | 0.0881 |

## Test Results
| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| GRU4Rec | 0.0580 | 0.0318 | 0.1000 | 0.0424 |
| SASRec | 0.0240 | 0.0101 | 0.0360 | 0.0133 |
| Time-Aware SASRec | 0.0800 | 0.0417 | 0.1360 | 0.0557 |

## Key Findings
- GRU4Rec is a stable sequential baseline.
- Vanilla SASRec underperforms under the current setup.
- Relative time-aware attention substantially improves Transformer-based sequential recommendation.
- Time-Aware SASRec outperforms both vanilla SASRec and GRU4Rec on validation and test sets.
