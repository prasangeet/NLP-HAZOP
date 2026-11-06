# Experimental Synthesis: NLP for Structured HAZOP Reasoning

**Status:** Experimental synthesis of the completed E1–E7 runs  
**Evaluation track:** T-301 reconstructed  
**Dataset:** 40 reconstructed scenario records  
**Dataset SHA-256:** `1c8713c2b6fbc9cdf2e22e64a307fdb64f073158a78679bdc23ad803a7d2725e`  
**Primary split:** Scenario-level 80/20 holdout, seed 42 (32 training scenarios, 8 held-out scenarios, 19 queries)  
**Label status:** Reconstructed row-level silver associations; not expert-validated causal ground truth

> **Interpretation boundary.** All reported predictive metrics measure agreement with consequence associations co-listed in reconstructed worksheet rows. They do not establish that a cause physically produces every co-listed consequence, nor do they establish operational safety or causal validity.

## 1. Executive summary

The experiments investigate whether typed graph reasoning can recover held-out cause-to-consequence worksheet associations more effectively than lexical retrieval, and how its behaviour changes under stricter evidence-support requirements, provenance-coherence filters, different scenario holdouts, and controlled removal of training relationships.

Across the fixed seed-42 comparison, graph reasoning achieved the strongest deterministic F1 (0.7442) and substantially outperformed TF–IDF retrieval (0.3553). The deterministic graph-plus-TF–IDF union increased coverage but reduced precision and F1 (0.5549). LLM conditions did not outperform graph reasoning on the matched subset and incurred much higher latency; the prompt-only condition almost always abstained.

The follow-on experiments qualify that result. Requiring repeated support for complete paths or individual cause edges sharply reduced recall and coverage. Requiring all path edges to share one scenario or source row also sharply reduced coverage without improving precision. The E6 repeated-holdout results show non-trivial split sensitivity. E7 demonstrates a monotonic decline in graph-reasoner recall, F1, and coverage as training relationships are randomly masked, while precision remains comparatively stable.

The evidence therefore supports a narrow conclusion: **on this reconstructed T-301 benchmark, typed graph reasoning is the strongest tested deterministic method for recovering row-derived silver associations, but its performance depends on the availability and composition of training relationships and does not establish causal correctness.**

## 2. Dataset and evaluation protocol

### 2.1 Dataset

The evaluated track contains 40 records reconstructed from the T-301 hexane-storage case reported in the literature. The records are not a verbatim transcription of a complete expert worksheet. The 54-row T-301 synthetic continuation and the 30-row DWSIM synthetic dataset are separate tracks and are not pooled into these E1–E7 results.

### 2.2 Query and target construction

The fixed benchmark creates one query per listed cause in a held-out scenario. The answer set consists of the consequences co-listed in that same row. All rows belonging to a scenario remain in one split. The held-out target record is stripped of causal labels and graph edges before the retrieval repository is built.

This is a leakage-aware test of recovering row associations from other scenarios. Because the source material does not independently adjudicate every cause–consequence pair, the target set is a silver label.

### 2.3 Metrics

- Micro and macro precision, recall, and F1 are computed against the row-derived silver target sets.
- Coverage is the fraction of eligible queries receiving at least one proposal; abstention is its complement.
- Selective risk is the error rate among answered queries and must be read together with coverage.
- MRR and Hit@1 measure ranking where applicable.
- Latency is implementation- and environment-dependent; it is not a hardware-independent performance guarantee.

## 3. E1 — Controlled strategy comparison

Protocol 1.3; seed 42; 32 training scenarios; 8 held-out scenarios; 19 test queries; top-k 10; maximum graph depth 4. The LLM conditions were evaluated on a deterministic cap of 15 cases. Full-test and matched-case results must not be conflated.

### 3.1 Full-test results

| Method | Evaluated / 19 | Micro precision | Micro recall | Micro F1 | Coverage | Abstention | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prompt-only LLM | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0667 | 0.9333 | 16,802 ms |
| TF–IDF retrieval | 19 | 0.2784 | 0.4909 | 0.3553 | 0.8947 | 0.1053 | 0.51 ms |
| Graph reasoning | 19 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 | 1.34 ms |
| Graph + TF–IDF union | 19 | 0.4068 | 0.8727 | 0.5549 | 0.8947 | 0.1053 | 1.85 ms |
| Hybrid graph/vector/LLM | 15 | 0.3704 | 0.6383 | 0.4688 | 0.8667 | 0.1333 | 38,415 ms |

The LLM methods did not evaluate all 19 cases. Their full-test rows therefore have different denominators and must not be treated as directly matched comparisons with the deterministic methods.

### 3.2 Matched 15-case comparison

| Method | Micro precision | Micro recall | Micro F1 | Coverage |
|---|---:|---:|---:|---:|
| Prompt-only LLM | 0.0000 | 0.0000 | 0.0000 | 0.0667 |
| TF–IDF retrieval | 0.3462 | 0.5745 | 0.4320 | 0.8667 |
| Graph reasoning | 0.6452 | 0.8511 | 0.7339 | 0.8000 |
| Graph + TF–IDF union | 0.4396 | 0.8511 | 0.5797 | 0.8667 |
| Hybrid graph/vector/LLM | 0.3704 | 0.6383 | 0.4688 | 0.8667 |

Graph reasoning remained strongest on the matched subset. The union preserved graph recall on this subset and increased coverage, but introduced additional false positives and reduced F1. The hybrid method did not improve on graph reasoning and was substantially slower in this run. Provider-call latency is highly environment- and service-dependent.

**Finding:** The graph reasoner is the strongest deterministic method in this experiment. The union is a coverage-oriented alternative, not a more accurate method. The tested LLM configurations did not demonstrate an advantage on this small benchmark.

## 4. E2 — Global complete-path support

E2 varied the minimum number of distinct eligible training scenarios supporting the same complete typed path:

`Cause → Deviation → Consequence`.

| Minimum supporting scenarios | Precision | Recall | F1 | Coverage | Abstention |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |
| 2 | 0.6000 | 0.1091 | 0.1846 | 0.3158 | 0.6842 |
| 3–6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

The unmodified threshold-1 condition reproduces the graph baseline. Requiring repeated support for the entire path sharply reduces coverage and recall. At thresholds of three or more, no eligible proposals remain.

**Finding:** Complete-path recurrence is sparse in this dataset. Requiring repeated observations of the entire path is too restrictive for useful recovery under the current evidence density. Distinct scenario counts are not independent physical confirmations; scenarios may share source assumptions.

## 5. E3 — Independent typed-edge support

E3 evaluated cause-to-deviation and deviation-to-consequence edge support independently. The two edges may be supported by different training scenarios.

| Condition | Precision | Recall | F1 | Coverage | Abstention |
|---|---:|---:|---:|---:|---:|
| Both edge thresholds 1 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |
| Cause edge ≥ 2 | 0.5938 | 0.3455 | 0.4368 | 0.3158 | 0.6842 |
| Consequence edge ≥ 2 | 0.6094 | 0.7091 | 0.6555 | 0.8421 | 0.1579 |
| Both edges ≥ 2 | 0.6000 | 0.3273 | 0.4235 | 0.3158 | 0.6842 |

The baseline is not expected to be identical to E2's path-support definition in general: E3 counts support for each typed edge independently, then joins qualifying edges. In this fixed dataset, the (1,1) result matches the baseline.

**Finding:** Cause-to-deviation support is the more restrictive bottleneck in this experiment. Raising its threshold reduces coverage and recall sharply. Raising only the consequence-edge threshold has a smaller effect. Independent edge support does not provide a free precision improvement.

## 6. E4 — Multi-hop reasoning

| Path condition | Candidate paths | Precision | Recall | F1 | Coverage | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 265 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |
| Intermediate multi-hop | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Direct + intermediate | 265 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |

No eligible intermediate multi-hop paths were available under the evaluated data and path constraints. Consequently, the combined condition reduces to direct-path behaviour.

**Finding:** E4 did not empirically exercise intermediate multi-hop reasoning. It is not evidence that longer causal paths are inherently ineffective; the evaluated graph did not supply eligible paths for that comparison.

## 7. E5 — Provenance coherence and selective acceptance

### 7.1 E5-1.0: metadata-presence thresholds

Thresholds 0, 0.5, 0.75, and 1 produced identical results:

| Precision | Selective risk | F1 | Coverage | Abstention | Unsupported | Traceable |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6486 | 0.3514 | 0.7442 | 0.8421 | 0.1579 | 0 | 1.0000 |

There were 74 accepted proposals: 48 true positives, 26 false positives, and 7 false negatives. Metadata completeness did not discriminate correct from incorrect proposals. Provenance-field presence is not equivalent to evidence strength.

### 7.2 E5-2.0: source-coherence filters

| Metric | Ungated | Same scenario | Same source row |
|---|---:|---:|---:|
| Precision | 0.6486 | 0.5000 | 0.5000 |
| Recall | 0.8727 | 0.1091 | 0.1091 |
| F1 | 0.7442 | 0.1791 | 0.1791 |
| Coverage | 0.8421 | 0.2105 | 0.2105 |
| Abstention | 0.1579 | 0.7895 | 0.7895 |
| Selective risk | 0.3514 | 0.5000 | 0.5000 |
| Accepted proposals | 74 | 12 | 12 |
| Rejected proposals | 0 | 62 | 62 |
| TP / FP / FN | 48 / 26 / 7 | 6 / 6 / 49 | 6 / 6 / 49 |

Both coherence filters behaved identically on this run. They rejected 62 of 74 proposals, retaining only 6 true positives and 6 false positives. Precision decreased rather than improved.

**Finding:** Provenance coherence is useful for traceability but cannot be treated as a causal-validity score. Strict same-source constraints can destroy useful cross-record evidence without resolving false positives.

## 8. E6 — Scenario-split robustness

E6 repeated the scenario-level 80/20 holdout for ten seeds (42–51). Each run held out 8 of the 40 scenarios; the test query count varied from 17 to 27. Across the ten runs, 36 unique scenarios appeared in a test split and 4 never did.

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Micro precision | 0.5765 | 0.0412 | 0.5294 | 0.6486 |
| Micro recall | 0.8615 | 0.1034 | 0.6383 | 1.0000 |
| Micro F1 | 0.6881 | 0.0479 | 0.5825 | 0.7442 |
| Macro F1 | 0.6266 | 0.0973 | 0.3722 | 0.7057 |
| Coverage | 0.8746 | 0.1283 | 0.5500 | 1.0000 |
| Abstention | 0.1254 | 0.1283 | 0.0000 | 0.4500 |

These are ten repeated random holdouts, **not ten-fold cross-validation**. The runs share much of their training data and are not independent observations. Four scenarios were never held out, so this design does not guarantee complete test coverage.

**Finding:** The graph-reasoner result is not identical across scenario partitions. Mean F1 remains useful as a descriptive summary, but split sensitivity is material and should accompany the single seed-42 result.

## 9. E7 — Controlled relationship-evidence masking

E7 used the fixed seed-42 scenario split and randomly removed typed `CausalRelation` objects from the 181 training relationships. It evaluated 10 masking seeds (42–51) at 10%, 20%, 30%, and 40%, for 40 conditions total. For each seed, masks are nested across rates. Test labels were not modified. The TF–IDF endpoint fields were reconstructed from the retained relations. LLMs were disabled.

### 9.1 Graph reasoning

| Training relationships masked | Mean precision | Mean recall | Mean F1 | Mean coverage | Mean abstention |
|---:|---:|---:|---:|---:|---:|
| 0% (E1 fixed-split baseline) | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |
| 10% | 0.6454 | 0.8109 | 0.7182 | 0.8000 | 0.2000 |
| 20% | 0.6439 | 0.7382 | 0.6872 | 0.7316 | 0.2684 |
| 30% | 0.6396 | 0.6582 | 0.6469 | 0.6842 | 0.3158 |
| 40% | 0.6476 | 0.5727 | 0.6030 | 0.6158 | 0.3842 |

Precision remains comparatively stable, while recall, F1, and coverage decline as more training relationships are removed.

### 9.2 Comparison methods

| Mask | Graph + TF–IDF union F1 | Union coverage | TF–IDF F1 | TF–IDF coverage |
|---:|---:|---:|---:|---:|
| 10% | 0.5432 | 0.8737 | 0.3288 | 0.8684 |
| 20% | 0.5305 | 0.8526 | 0.3143 | 0.8368 |
| 30% | 0.5139 | 0.8263 | 0.2775 | 0.8053 |
| 40% | 0.4936 | 0.7737 | 0.2402 | 0.7526 |

The union has higher coverage than graph reasoning, but its F1 is lower at every masking rate. TF–IDF is the weakest method in this controlled masking experiment.

**Finding:** Graph reasoning degrades as available training relationships are removed, primarily through lower recall and more abstention. This is evidence of sensitivity to simulated evidence loss, not a measurement of naturally occurring missingness.

## 10. Cross-experiment synthesis

The experiments collectively support the following observations:

1. **Typed graph reasoning is the strongest tested deterministic approach on the fixed reconstructed T-301 benchmark.** It outperforms TF–IDF on micro F1 and maintains a higher precision–recall balance.
2. **Candidate union trades precision for coverage.** Adding lexical candidates recovers more silver targets in some cases but introduces false positives; it is not a learned fusion model.
3. **Evidence recurrence is sparse.** Strict repeated-path and cause-edge support thresholds lead to substantial abstention and recall loss.
4. **Provenance completeness is not causal validation.** Metadata-only thresholds do not distinguish correct from incorrect predictions; strict same-source coherence can over-filter.
5. **Intermediate multi-hop reasoning remains untested by available paths.** E4 did not contain eligible intermediate paths.
6. **Results depend on the scenario split.** E6 shows a range of F1 and coverage across repeated holdouts.
7. **The graph reasoner depends on available relationship evidence.** E7 shows a monotonic reduction in average F1 and coverage under increasing random masking.

These observations should not be collapsed into a claim that graph reasoning is universally superior, causally correct, or safe for operational use.

## 11. Limitations and validity threats

### Dataset and label validity
- T-301 rows are reconstructed from a reported case, not a verbatim complete source worksheet.
- The labels are row co-occurrences. A cause and consequence appearing in the same row are not independently verified as a direct causal pair.
- No relationship-level expert-adjudicated gold set is available in the evaluated track.
- Synthetic DWSIM and T-301 continuation data must remain separate from reconstructed T-301 findings.

### Sample size and split uncertainty
- The fixed test set contains only 8 held-out scenarios and 19 queries.
- E6 comprises repeated holdouts, not independent folds; four scenarios were never held out.
- The experiments are diagnostic and exploratory. The observed metric differences should not be interpreted as broad statistical guarantees.

### Method and implementation scope
- E4's intermediate multi-hop condition had zero eligible paths.
- E5's source-coherence rules are coarse proxies for provenance consistency, not causal evidence evaluation.
- E7 uses random masking of loader-generated typed relations. It does not reproduce the mechanisms that create missingness in real HAZOP documentation.
- The LLM evaluation was capped at 15 cases, and cost was not reliably captured. Latency is specific to the recorded environment and provider state.
- Structural validation checks do not replace process-model validation, physical plausibility checks, or expert review.

### Software scope
- The repository contains a working research prototype architecture and test suite, but a complete application workflow must be validated in the target local environment before claiming end-to-end readiness.
- PDF text extraction does not imply OCR support for scanned/image-only PDFs.
- Neo4j persistence requires correctly configured credentials and a reachable service; preview-only functionality can operate without it.

## 12. Remaining work before research submission

The following are genuine remaining requirements, not additional experiments to run automatically:

1. **Expert adjudication:** Obtain independent review of a held-out subset using a written rubric for causal plausibility, context sufficiency, and safeguard validity. Record reviewer disagreements and inter-rater agreement. Until then, causal correctness remains unverified.
2. **Grouped cross-validation (optional but valuable):** Integrate the existing grouped split primitive into a full runner that rebuilds each fold's index from training data and evaluates identical held-out cases. Do not call E6 cross-validation.
3. **End-to-end application validation:** Run the backend test suite, frontend type/build checks, and a manual Tauri workflow covering dataset preview/import, graph inspection, recovery, report extraction, and saved experiment results.
4. **Reproducibility audit:** Confirm experiment scripts, frozen JSON/JSONL artifacts, dataset hash, and commit identifiers are preserved. Record the exact local environment and dependencies used for the final runs.
5. **Final report:** Use this synthesis as the results/discussion basis. Keep the literature review, methods, results, limitations, and conclusions aligned; remove obsolete roadmap statements and stale baseline descriptions.
6. **Repository hygiene:** Review untracked and modified local files before committing. Keep raw frozen experiment outputs immutable and avoid committing secrets, temporary files, caches, or large generated binaries unnecessarily.

## 13. Defensible conclusion

On the evaluated reconstructed T-301 dataset, graph-based reasoning recovers held-out row-derived consequence associations more accurately than the tested TF–IDF and LLM configurations. Its performance is sensitive to the amount and distribution of available training relationships. More restrictive recurrence and provenance-coherence filters reduce coverage substantially and do not yield a precision improvement in these experiments. The deterministic graph-plus-retrieval union increases coverage but lowers precision and F1.

The experiments demonstrate the value of explicitly representing typed HAZOP relationships for structured retrieval and reasoning over this dataset. They do **not** establish that recovered relationships are physically causal, expert-approved, or safe to use in plant operations. Establishing those claims requires independent process-safety adjudication and stronger validation data.
