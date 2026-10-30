# Dataset provenance and handling

## T-301 hexane storage tank

File: data/raw/t301/T301_HAZOP_CCPS2001_style.csv

This file contains 94 records:

- 40 PAPER_REPORTED_RECONSTRUCTION records: structured reconstructions of scenario chains reported for the T-301 hexane storage case in Single et al. (2020). They are not verbatim source rows.
- 54 SYNTHETIC_CONTINUATION records: project-generated continuation scenarios. They must not be attributed to the published paper.

The benchmark exposes these as two separate tracks: t301-reconstructed and t301-synthetic.

## DWSIM ethanol/fermentation plant

Canonical benchmark file: data/raw/dwsim/synthetic_hazop_D001_D030.csv

The D001–D030 records are synthetic and simulation/inference-derived. They include model/review caveats. The supplied status text includes:

- 16 Review required
- 6 Scope caveat
- 6 NEEDS_MODEL_REVISION
- 1 Review reaction model
- 1 NEEDS_REVIEW

None of these labels means the record has passed expert validation.

The file data/raw/dwsim/HAZOP_DWSIM30_worksheet_format.csv is a worksheet-style export of the same D001–D030 scenarios. It is an alternate representation, not an independent second DWSIM sample.

## Provenance rules

1. Never label reconstructed or synthetic records as source-verified.
2. Never label a record expert-validated unless an explicit expert validation event and reviewer provenance exist.
3. Cause/deviation/effect/consequence edges derived from row fields are row-level associations. They are useful for testing graph construction but are not independent evidence of causation.
4. Retrieval-generated triples carry RETRIEVAL_INFERRED provenance. Multi-hop path outputs carry GRAPH_INFERRED provenance.
5. Model-only proposals carry MODEL_INFERRED provenance.
6. Keep dataset tracks separate in splits, metrics, figures, and conclusions.

## Benchmark target definition

The current evaluator holds out entire scenarios. For each held-out cause, it uses the consequences co-listed in that same row as a silver candidate set. Since source rows do not provide adjudicated cause-to-consequence mappings, this label definition cannot support strong causal correctness claims. Expert adjudication is a required future step.
