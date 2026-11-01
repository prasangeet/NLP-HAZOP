# HAZOP Recovery Benchmark Report

> Evaluation uses silver row-association labels. It does not establish causal validity or safety certification.

## Run metadata

- Dataset: t301-reconstructed
- Dataset SHA-256: 1c8713c2b6fbc9cdf2e22e64a307fdb64f073158a78679bdc23ad803a7d2725e
- Repository commit: 32cac98c68abfb00372a2ce0da2946aeb4b93714
- Protocol version: 1.0
- Seed / test fraction: 42 / 0.2
- Scenarios (train/test): 32 / 8
- Test queries: 19
- Created at: 2026-11-01T17:04:58.320760+00:00

## Method results

| Method | Status | Evaluated / total | Micro F1 | Coverage | Abstention | Mean latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| prompt_only_llm | disabled_or_call_cap_zero | 0 / 19 | not reported | not reported | not reported | not reported |
| tfidf_vector_retrieval | evaluated | 19 / 19 | 0.3421 | 0.8947 | 0.1053 | 0.1905 |
| graph_reasoning | evaluated | 19 / 19 | 0.0000 | 0.0000 | 1.0000 | 1.3918 |
| graph_tfidf_union | evaluated | 19 / 19 | 0.3421 | 0.8947 | 0.1053 | 1.5823 |
| hybrid_graph_vector_llm | disabled_or_call_cap_zero | 0 / 19 | not reported | not reported | not reported | not reported |

## Matched-case comparison

- Comparison subset size: 15
- Selection: first N cases in deterministic scenario/cause order; N equals max_llm_cases

| Method | Status | Evaluated / matched | Micro F1 | Coverage |
|---|---|---:|---:|---:|
| prompt_only_llm | not_evaluated_on_comparison_cases | 0 / 15 | not reported | not reported |
| tfidf_vector_retrieval | evaluated | 15 / 15 | 0.4160 | 0.8667 |
| graph_reasoning | evaluated | 15 / 15 | 0.0000 | 0.0000 |
| graph_tfidf_union | evaluated | 15 / 15 | 0.4160 | 0.8667 |
| hybrid_graph_vector_llm | not_evaluated_on_comparison_cases | 0 / 15 | not reported | not reported |

## Interpretation notes

- Missing or disabled methods are not scored as zero; inspect their status.
- F1 is agreement with silver row associations, not verified causal ground truth.
- Compare methods only when evaluated on identical case IDs.
- See results.json for metadata and per-case outputs, and predictions.csv for prediction-level rows.
