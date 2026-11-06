# E3 Case-Level Error Analysis

Baseline: `edge_support_1_1`.

## Comparison summary

| Condition | Cases losing gold | Gold lost | Gold newly recovered | Predictions removed | Predictions added | Became abstentions |
|---|---:|---:|---:|---:|---:|---:|
| `cause_consensus_2_1` | 10 | 29 | 0 | 42 | 0 | 10 |
| `consequence_consensus_1_2` | 5 | 9 | 0 | 10 | 0 | 0 |
| `both_edges_consensus_2_2` | 11 | 30 | 0 | 44 | 0 | 10 |

## Saved metric consistency

| Condition | Precision | Recall | F1 | Coverage | Abstention |
|---|---|---|---|---|---|
| `edge_support_1_1` | match | match | match | match | match |
| `cause_consensus_2_1` | match | match | match | match | match |
| `consequence_consensus_1_2` | match | match | match | match | match |
| `both_edges_consensus_2_2` | match | match | match | match | match |

## Interpretation cautions

- A lost gold value means it was predicted in the (1,1) condition but not in the comparison condition.
- A removed prediction is not automatically a false positive; it may be a correct gold prediction or an incorrect prediction.
- Aggregate qualifying-edge counts do not reveal the support count of the exact edge associated with an individual gold value.
- This analysis is based on reconstructed row-level silver labels, not expert-validated causal ground truth.
- Scenario recurrence does not establish independent evidence or physical causality.
