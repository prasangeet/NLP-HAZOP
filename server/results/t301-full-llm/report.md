# HAZOP Recovery Benchmark Report

> Evaluation uses silver row-association labels. It does not establish causal validity or safety certification.

## Run metadata

- Dataset: t301-reconstructed
- Dataset SHA-256: 1c8713c2b6fbc9cdf2e22e64a307fdb64f073158a78679bdc23ad803a7d2725e
- Repository commit: 04da44fcaa593fea920f91eb0856bd3dc01567b4
- Protocol version: 1.2
- Seed / test fraction: 42 / 0.2
- Scenarios (train/test): 32 / 8
- Test queries: 19
- Created at: 2026-11-01T22:56:25.905455+00:00

## Method results

| Method | Status | Evaluated / total | Micro F1 | Coverage | Abstention | Mean latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| prompt_only_llm | evaluated | 15 / 19 | 0.0000 | 0.2000 | 0.8000 | 5758.6982 |
| tfidf_vector_retrieval | evaluated | 19 / 19 | 0.3421 | 0.8947 | 0.1053 | 0.2494 |
| graph_reasoning | evaluated | 19 / 19 | 0.7442 | 0.8421 | 0.1579 | 1.8163 |
| graph_tfidf_union | evaluated | 19 / 19 | 0.5455 | 0.8947 | 0.1053 | 2.0657 |
| hybrid_graph_vector_llm | evaluated | 15 / 19 | 0.5674 | 0.8667 | 0.1333 | 16486.0552 |

## Matched-case comparison

- Comparison subset size: 15
- Selection: first N cases in deterministic scenario/cause order; N equals max_llm_cases

| Method | Status | Evaluated / matched | Micro F1 | Coverage |
|---|---|---:|---:|---:|
| prompt_only_llm | evaluated | 15 / 15 | 0.0000 | 0.2000 |
| tfidf_vector_retrieval | evaluated | 15 / 15 | 0.4160 | 0.8667 |
| graph_reasoning | evaluated | 15 / 15 | 0.7339 | 0.8000 |
| graph_tfidf_union | evaluated | 15 / 15 | 0.5674 | 0.8667 |
| hybrid_graph_vector_llm | evaluated | 15 / 15 | 0.5674 | 0.8667 |

## Interpretation notes

- Missing or disabled methods are not scored as zero; inspect their status.
- F1 is agreement with silver row associations, not verified causal ground truth.
- Compare methods only when evaluated on identical case IDs.
- See results.json for metadata and per-case outputs, and predictions.csv for prediction-level rows.
