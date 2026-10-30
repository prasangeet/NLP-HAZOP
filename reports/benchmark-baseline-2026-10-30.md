# HAZOP benchmark baseline — 30 October 2026

<sup>Exploratory baseline run; not an expert-validated safety evaluation.</sup>

## Run provenance

- Repository commit: \`8f565e644306bd2e913a36565f5fbade5771e204\`
- GitHub Actions run: [41048488263](https://github.com/prasangeet/NLP-HAZOP/actions/runs/41048488263)
- Validation: Ruff passed; **122 tests passed**; all three dataset benchmark jobs passed.
- Split: scenario-level 80/20 holdout, seed 42.
- Candidate limit: top 10; graph search cap: 4 hops.
- Query construction: one query per cause listed in each held-out row; the silver answer set is the set of consequences co-listed in that same row.

The labels measure agreement with row-level co-listings. They do not establish that every listed cause produces every listed consequence. The supplied datasets are not expert-validated.

## Dataset split

| Dataset track | Total rows | Train scenarios | Test scenarios | Test queries |
|---|---:|---:|---:|---:|
| T-301 reconstructed | 40 | 32 | 8 | 19 |
| T-301 synthetic continuation | 54 | 43 | 11 | 11 |
| DWSIM synthetic | 30 | 24 | 6 | 15 |

The three tracks were evaluated independently and were not pooled.

## Results

### Sparse TF–IDF retrieval

| Metric | T-301 reconstructed | T-301 synthetic | DWSIM synthetic |
|---|---:|---:|---:|
| Micro precision | 26.80% | 0.00% | 9.00% |
| Micro recall | 47.27% | 0.00% | 34.62% |
| Micro F1 | 34.21% | 0.00% | 14.29% |
| Macro F1 | 29.68% | 0.00% | 13.43% |
| Hit@1 | 15.79% | 0.00% | 26.67% |
| MRR | 0.3456 | 0.0000 | 0.3278 |
| Coverage | 89.47% | 100.00% | 93.33% |
| Abstention rate | 10.53% | 0.00% | 6.67% |
| Mean candidates/query | 5.11 | 7.09 | 6.67 |
| Mean query latency | 0.151 ms | 0.371 ms | 0.131 ms |
| Index build time | 0.754 ms | 1.330 ms | 0.551 ms |

All returned TF–IDF candidates carry \`retrieval_inferred\` provenance. The measured source-or-expert origin rate is therefore 0%; these candidates are row analogies, not direct source-verified edges.

### Graph reasoning

| Metric | T-301 reconstructed | T-301 synthetic | DWSIM synthetic |
|---|---:|---:|---:|
| Micro precision | 0.00% | 0.00% | 0.00% |
| Micro recall | 0.00% | 0.00% | 0.00% |
| Micro F1 | 0.00% | 0.00% | 0.00% |
| Coverage | 0.00% | 0.00% | 0.00% |
| Abstention rate | 100.00% | 100.00% | 100.00% |

Abstention reasons:
- T-301 reconstructed: 17 ambiguous-evidence abstentions, 2 insufficient-support abstentions.
- T-301 synthetic: 11 insufficient-support abstentions.
- DWSIM synthetic: 15 insufficient-support abstentions.

The graph path strategy is conservative and did not make a proposal on this split. In the reconstructed T-301 track, repeated cause nodes lead to competing terminal paths; the reasoner abstained rather than choose among them. In the synthetic tracks, the held-out cause nodes generally had no supported training path. This is a current retrieval/coverage limitation, not evidence that graph reasoning is intrinsically ineffective.

### LLM methods

Prompt-only LLM and hybrid graph + vector + LLM were **not evaluated** in this run. The default CI command does not enable LLM calls and no Groq credential was supplied to the workflow. Their metrics are intentionally absent, not zero.

## Interpretation

1. On reconstructed T-301 silver labels, TF–IDF retrieved some co-listed consequences, with micro F1 0.3421 and coverage 89.47%.
2. On synthetic T-301, TF–IDF returned candidates but none matched the held-out row silver sets. High coverage alone is not evidence of correctness.
3. On DWSIM synthetic, TF–IDF recovered a portion of the row-associated labels (micro F1 0.1429), but precision was low (0.09) and the labels themselves are synthetic and unreviewed.
4. Graph reasoning abstained on all queries. Its ambiguity abstentions in reconstructed T-301 should be retained as a meaningful safety behavior, but graph-context matching and coverage need investigation.
5. No method has yet been evaluated against expert-adjudicated causal relations. These results must not be presented as validated causal correctness or operational HAZOP performance.

## Remaining work before research conclusions

1. Run prompt-only and hybrid LLM methods on a controlled, identical held-out query subset with a configured Groq key; record model/version, request failures, token usage, and cost.
2. Improve graph retrieval coverage without weakening provenance or forcing a selection when paths conflict. Evaluate exact-node matching, scenario attributes, and ontology-level entity normalization as separate ablations.
3. Obtain expert adjudication for a held-out subset of cause–deviation–effect–consequence links. Record reviewer identity/role, adjudication criteria, disagreements, and inter-rater agreement.
4. Repeat the split with multiple seeds or scenario-level cross-validation and report uncertainty intervals.
5. Add an explicit comparison against a dense embedding retriever only if that method is part of the intended experimental claim; the current vector baseline is sparse TF–IDF.
6. Capture provider token usage and cost for LLM runs; current \`cost_usd\` is null.

## Artifact files

GitHub Actions uploaded a separate ZIP artifact for each dataset track under run 41048488263. Each contains \`results.json\`, \`summary.csv\`, and \`predictions.csv\`.
