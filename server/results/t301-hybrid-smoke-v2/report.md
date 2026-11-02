# HAZOP Recovery Benchmark Report

> Evaluation uses silver row-association labels. It does not establish causal validity or safety certification.

## Run metadata

- Dataset: t301-reconstructed
- Dataset SHA-256: 1c8713c2b6fbc9cdf2e22e64a307fdb64f073158a78679bdc23ad803a7d2725e
- Repository commit: 50923a9f029e9828fa7baecfed364abeba5cec54
- Protocol version: 1.2
- Seed / test fraction: 42 / 0.2
- Scenarios (train/test): 32 / 8
- Test queries: 19
- Created at: 2026-11-01T23:54:17.356845+00:00

## Method results

| Method | Status | Evaluated / total | Micro F1 | Coverage | Abstention | Mean latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| prompt_only_llm | evaluated | 3 / 19 | 0.0000 | 0.0000 | 1.0000 | 3565.4978 |
| tfidf_vector_retrieval | evaluated | 19 / 19 | 0.3421 | 0.8947 | 0.1053 | 0.3058 |
| graph_reasoning | evaluated | 19 / 19 | 0.7442 | 0.8421 | 0.1579 | 2.5318 |
| graph_tfidf_union | evaluated | 19 / 19 | 0.5455 | 0.8947 | 0.1053 | 2.8376 |
| hybrid_graph_vector_llm | evaluated | 3 / 19 | 0.7059 | 1.0000 | 0.0000 | 9055.4678 |

## Matched-case comparison

- Comparison subset size: 3
- Selection: first N cases in deterministic scenario/cause order; N equals max_llm_cases

| Method | Status | Evaluated / matched | Micro F1 | Coverage |
|---|---|---:|---:|---:|
| prompt_only_llm | evaluated | 3 / 3 | 0.0000 | 0.0000 |
| tfidf_vector_retrieval | evaluated | 3 / 3 | 0.4828 | 1.0000 |
| graph_reasoning | evaluated | 3 / 3 | 0.8889 | 1.0000 |
| graph_tfidf_union | evaluated | 3 / 3 | 0.7059 | 1.0000 |
| hybrid_graph_vector_llm | evaluated | 3 / 3 | 0.7059 | 1.0000 |

## Interpretation notes

- Missing or disabled methods are not scored as zero; inspect their status.
- F1 is agreement with silver row associations, not verified causal ground truth.
- Compare methods only when evaluated on identical case IDs.
- See results.json for metadata and per-case outputs, and predictions.csv for prediction-level rows.
