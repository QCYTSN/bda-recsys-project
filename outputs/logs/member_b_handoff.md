# Member B Handoff

## Implemented Models
- GRU4Rec
- SASRec
- Time-Aware SASRec

## Best Sequence Model
- Time-Aware SASRec

## Sequence Representation
For sequence models, the sequence representation `E_seq` is defined as the hidden state at the last non-padding position of the input sequence.

## Current Dimension
- hidden_dim = 64

## Current Validation Results
| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| GRU4Rec | 0.0980 | 0.0603 | 0.1680 | 0.0781 |
| SASRec | 0.0200 | 0.0082 | 0.0460 | 0.0147 |
| Time-Aware SASRec | 0.1320 | 0.0702 | 0.2040 | 0.0881 |

## Key Finding
Relative time-aware attention substantially improves Transformer-based sequential recommendation on the current Amazon Electronics setup.
