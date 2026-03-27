# Baseline Validation Results

## Setting
- Dataset: Amazon Electronics
- Implicit feedback: rating >= 3
- Max sequence length: 50
- Epochs: 3
- Validation users: 500
- Metrics: HR@10, NDCG@10
- Seen-item masking: enabled

## Results
| Model | HR@10 | NDCG@10 |
|---|---:|---:|
| GRU4Rec | 0.1000 | 0.0507 |
| SASRec | 0.0200 | 0.0082 |
| Time-Aware SASRec | 0.1280 | 0.0663 |

## Observations
- GRU4Rec is a strong baseline under the current setting.
- Vanilla SASRec performs poorly on this dataset/setup.
- Adding relative time-aware attention bias substantially improves Transformer-based sequential recommendation.
- Time-Aware SASRec outperforms both vanilla SASRec and GRU4Rec in the current validation setting.
