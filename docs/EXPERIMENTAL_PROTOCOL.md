# Experimental Protocol: NLP for Structured HAZOP Reasoning

**Status:** E1–E7 completed for the reconstructed T-301 track; expert adjudication and end-to-end application validation remain outstanding.  
**Purpose:** Define the evidence required to evaluate missing-link recovery and structured causal reasoning without overstating what the current datasets establish.

## 1. Research claims and falsifiable questions

### RQ1 — Recovery against held-out associations
Can a recovery method predict held-out consequence labels from the available cause and scenario context better than simple retrieval baselines?

**H1:** At least one structured or hybrid method improves macro/micro F1 over the strongest non-LLM baseline on held-out scenarios. This hypothesis is unsupported until identical test cases have been evaluated.

### RQ2 — Value of explicit graph structure
Does graph evidence improve ranking or reduce unsupported predictions relative to lexical retrieval and prompt-only LLM inference?

**H2:** Graph-assisted methods improve held-out recovery quality or evidence traceability over the relevant baseline, without increasing unsupported proposals. Compare on identical cases and report coverage/abstention alongside accuracy.

### RQ3 — Abstention under ambiguity
Can the reasoner abstain on ambiguous or insufficient evidence while maintaining useful coverage?

**H3:** At a predeclared coverage level, a provenance-aware method has a lower error rate among answered cases than a non-abstaining baseline. Report the coverage–risk curve; do not treat high abstention by itself as success.

### RQ4 — Generalization
Does performance transfer between reconstructed T-301 material and DWSIM-derived synthetic scenarios?

This is exploratory until dataset provenance and domain differences are documented. Do not pool these tracks by default.

## 2. Current benchmark and its limits

The existing `server/scripts/run_benchmark.py` is the starting point. The recorded baseline uses scenario-level 80/20 holdout with seed 42 and generates one query per listed cause. The row's co-listed consequences are used as silver targets.

Current tracks:
- T-301 reconstructed: 40 rows, 19 test queries.
- T-301 synthetic continuation: 54 rows, 11 test queries.
- DWSIM synthetic: 30 rows, 15 test queries.

The T-301 reconstructed track is not a verbatim source dataset; the synthetic tracks are not expert-validated. Row co-occurrence is not proof of causality. Metrics therefore measure agreement with **silver row associations**, not true causal validity or operational safety.

Recorded 30 October 2026 baseline:
- TF-IDF micro F1: 34.21% on T-301 reconstructed, 0.00% on T-301 synthetic, 14.29% on DWSIM synthetic.
- Graph reasoner coverage: 0% on all three tracks, with 100% abstention.
- Prompt-only LLM and hybrid LLM were not evaluated in that run.

These values are a historical baseline to reproduce, not proof of the research hypothesis.

## 3. Required controls

1. **Identical cases:** Compare methods on exactly the same held-out query IDs.
2. **Scenario-level isolation:** No scenario, target consequence, or target-derived feature from test scenarios may enter training graphs, retrievers, normalization dictionaries, prompts, or tuning.
3. **Fit on training only:** Any learned vocabulary, embedding index, threshold, canonicalization map, or hyperparameter must be fit/tuned on training/validation data only.
4. **Predeclare target:** The initial task is cause-to-consequence set recovery. Do not silently mix effects, safeguards, deviations, and consequences into one metric.
5. **Provenance:** Preserve whether each prediction comes from source evidence, expert annotation, graph inference, retrieval analogy, or LLM generation.
6. **Failures count:** Record provider errors, timeouts, malformed responses, and skipped cases; never silently drop failed cases from reported denominators.
7. **Reproducibility:** Record repository commit, dataset version/hash, split seed, configuration, model/provider/version, prompt version, timestamp, runtime environment, and raw per-case outputs.
8. **No fabricated scores:** A missing method/result is “not run,” not zero. Do not report precision/recall/F1 without reference labels.
9. **Uncertainty:** Current test sets are small. Add repeated scenario-level seeds or grouped cross-validation and report per-seed results plus confidence intervals where appropriate.
10. **No causal overclaim:** Separate silver-label retrieval performance from expert-adjudicated causal correctness.

### Scenario-grouped cross-validation foundation

The benchmark module now provides `grouped_kfold_split(records, n_splits, seed)`.
It assigns unique `scenario_id` groups to deterministic folds, returns one
training/held-out pair per fold, and ensures every scenario appears in exactly
one held-out fold. All rows belonging to a scenario remain together.

This is currently a tested split primitive, **not yet an end-to-end
cross-validation experiment**. The existing CLI, API suite, and frontend still
execute the established single holdout per seed. Do not describe current
benchmark-suite results as cross-validation results.

Before using grouped folds for reported model metrics, integrate fold execution
so that each fold rebuilds its retrieval/graph index from that fold's training
records, strips held-out target labels and graph edges, and evaluates all
methods on identical held-out case IDs. LLM call caps must be applied within
each fold and reported with matched-case denominators. Report fold-level
metrics and the number of scenarios and queries per fold. Because folds share
overlapping training data and the current datasets are small, fold scores are
not independent observations; confidence intervals across folds must be
labelled descriptive rather than treated as independent-sample uncertainty.

## 4. Metrics

For set-valued consequence predictions, compute micro and macro precision, recall, and F1 against the declared target set. Also report:
- Hit@1 and mean reciprocal rank (MRR) for ranked candidates.
- Coverage = queries with at least one proposal / all eligible queries.
- Abstention rate and reason categories.
- Selective risk = error rate among queries answered, reported alongside coverage.
- Mean, median, and percentile latency; LLM token use and estimated cost where available.
- Provenance/evidence traceability rate and unsupported-proposal rate, with operational definitions fixed before evaluation.

State exact-match normalization rules. Preserve raw strings alongside normalized comparison values.

## 5. Completed experiment sequence

The following sequence describes the experiments actually completed for the
T-301 reconstructed track. It supersedes the earlier proposed E2–E7 labels
below; the experiment identifiers refer to the frozen artifacts in
`server/results/`.

| Experiment | Actual question | Status / artifact |
|---|---|---|
| E1 | Compare prompt-only LLM, TF–IDF, graph reasoning, graph+TF–IDF union, and hybrid reasoning on a fixed scenario holdout. | Completed; `server/results/t301-reconstructed/results.json` |
| E2 | Measure the effect of minimum distinct-scenario support for a complete typed cause–deviation–consequence path. | Completed; `server/results/e2/global_path_consensus.json` |
| E3 | Measure cause-edge and consequence-edge support thresholds independently. | Completed; `server/results/e3/edge_support_consensus.json` |
| E4 | Test direct versus intermediate multi-hop path reasoning under the available graph evidence. | Completed; `server/results/e4/multi_hop_reasoning.json` |
| E5 | Test metadata-presence thresholds and same-scenario / same-source-row provenance-coherence filters. | Completed; `server/results/e5/abstention_selective_risk_v2.json` |
| E6 | Measure sensitivity to ten repeated scenario-level 80/20 holdouts (seeds 42–51). | Completed; `server/results/e6/scenario_split_robustness_20261106T143828Z.json` |
| E7 | Measure sensitivity to random masking of 10–40% of typed training relationships over ten masking seeds. | Completed; `server/results/e7/e7_summary_20261106T151612Z.json` and the paired JSONL runs file |

### Interpretation constraints

- E6 is repeated random holdout, **not** grouped k-fold cross-validation.
  The ten runs share overlapping training data, and four of the 40 scenarios
  were never held out.
- E7 is controlled random evidence masking on loader-generated relationships.
  It is not a study of naturally occurring missingness.
- E1–E7 use reconstructed row-level silver associations for this track.
  None of these experiments establishes expert-adjudicated causal truth.
- E4's intermediate multi-hop condition produced no eligible intermediate
  paths; it therefore does not establish that longer causal chains are
  ineffective.
- E5's provenance filters test documentary coherence, not physical causality.
- E1 LLM methods were evaluated on a capped subset. Full-test and matched-case
  metrics must remain separate.
- Do not overwrite the frozen artifacts when preparing derived tables or
  figures. Derived summaries should cite the exact source artifact.

## 6. Experimental records and artifacts

Each run should write:
- `run_metadata.json`: commit, timestamps, dataset/hash, split, seed, configuration, methods, model/prompt versions, environment, dependency versions, token/cost metadata where applicable.
- `case_results.jsonl` or CSV: case ID, gold silver labels, ranked predictions, normalized predictions, correctness, abstention reason, provenance/evidence, latency, errors.
- `metrics.json`: metric definitions, denominators, per-dataset/per-method summaries, coverage, selective risk and uncertainty where computed.
- `report.md`: readable interpretation, caveats, deviations from protocol and comparison to prior runs.

Keep raw outputs immutable for each run. Derived summaries may be regenerated from raw case results.

## 7. Experiment-workbench requirements

The frontend should be a client for the real benchmark runner/API, not a simulated experiment engine and not merely an API health dashboard.

Required controls:
- Choose an available dataset track, experiment, methods, seed/split and supported configuration.
- Validate prerequisites and show unavailable methods as unavailable (for example, missing LLM credentials), rather than returning fake metrics.
- Start a run, show its real status, and display errors/failures.
- Present per-method/per-dataset metrics with denominators and coverage/abstention.
- Inspect per-case predictions, reference labels, evidence, provenance and abstention reason.
- Compare saved runs and download the configuration, raw case results and report.
- Mark silver-label and expert-adjudicated evaluations distinctly.
- Never display a metric as causal validity or safety certification.

## 7.1 Implemented automated ablation and repeated-seed summaries

The suite now accepts graph hop limits (2–8) alongside dataset tracks and
random seeds. Each selected combination runs the same benchmark CLI with
the requested `--max-hops` value. This is a **graph-depth sensitivity
ablation**, not a complete component ablation: it tests the effect of the
reasoner's maximum traversal depth while leaving the dataset split protocol,
retrieval settings, and other model components unchanged.

For each dataset, hop setting, method, and metric, the API aggregates
seed-level results. With two to six successful seeds it reports the sample
mean, sample standard deviation, and a two-sided 95% Student-t interval.
With one successful seed, uncertainty is explicitly marked not estimable.
The frontend displays the across-seed micro-F1 interval and mean coverage.
Intervals are descriptive of random split variability; they do not account
for dataset sampling bias, label uncertainty, or expert disagreement.

The frontend also retrieves each persisted run's Markdown report and CSV
artifacts from the API. A suite is limited to 18 dataset × seed × hop runs
to bound synchronous API work.

## 8.1 Automated frontend execution and persisted runs

The Experiments workbench exposes a multi-dataset, multi-seed suite through
`POST /api/v1/experiments/benchmark-suite`. The frontend submits selected
dataset tracks, unique random seeds, test fraction, and the optional LLM flag.
Each dataset/seed pair invokes the same benchmark CLI used by the single-run
endpoint. A failed pair is recorded in the suite manifest and does not silently
disappear from the run matrix.

Completed run artifacts are persisted under
`server/data/experiments/runs/<run_id>/<dataset>/seed-<seed>/`. The suite
manifest records the selected matrix, protocol version, label caveat, completed
runs, and failures. The frontend displays the returned per-method summaries
and refreshes the saved-run listing. Artifact retrieval is exposed through the
benchmark artifact API; generated run data is local to the API server and
should be backed up with the project data directory.

This automation currently covers repeated-seed execution of the TF-IDF,
graph, deterministic graph-plus-TF-IDF, and optional prompt-only / hybrid LLM
benchmark implementations. The deterministic graph-plus-TF-IDF condition (`graph_tfidf_union`)
unions the graph reasoner's proposal with TF-IDF-ranked retrieval candidates,
deduplicating normalized endpoint strings. Retrieval-inferred candidates are
not re-labelled as graph facts or passed through graph-path validation. This
tests complementary candidate-source coverage without LLM adjudication; it is
not a learned fusion or reranking method.

The LLM conditions retain the existing evidence constraints: proposals must
select exactly one supplied evidence item, satisfy the query direction and
predicate, and preserve provenance distinctions. These constraints are held
fixed as safety/validity guardrails; an unconstrained hallucination-prone LLM
variant is intentionally not included as an ablation condition. Prompt-only
LLM remains a separate scenario-context control, not a component of the
retrieval pipeline.

### LLM-first recovery path (protocol 1.2)

The hybrid recovery path now has two explicit LLM modes:

1. When retrieval or graph traversal supplies candidates, the evidence-grounded
   LLM adjudicator selects one supported candidate or abstains. It cannot
   invent or rewrite an evidence edge.
2. When neither retrieval nor graph traversal supplies a candidate, the
   hybrid strategy invokes a generative LLM fallback with the scenario context
   and available graph neighborhood. This path can propose a new endpoint;
   its origin is `model_inferred`, and it is not presented as a source-backed
   fact. The prompt-only benchmark condition remains scenario-context-only,
   while the hybrid fallback may additionally receive graph context.

All generated proposals are checked against the requested predicate, known
endpoint direction, dataset scope, ontology endpoint types, and the selected
HAZOP record's structured fields. A candidate that conflicts with the record
is rejected. These deterministic checks establish structural consistency
only; they do not establish physical plausibility, causal truth, or safety
approval. The proposal retains a review-required warning.

The workbench enables LLM evaluation by default. If LLM evaluation is requested
without a configured Groq API key, the API returns an explicit service error
rather than silently presenting the LLM methods as evaluated. The benchmark
records the model, call cap, and a maximum of two LLM calls per matched case
(one prompt-only call and one hybrid call).

This is a first component comparison, not a complete factorial ablation.
Grouped cross-validation, expert-adjudication workflow, and a formal audit of
all possible feature/ontology leakage remain pending. A suite button does not
establish experimental validity by itself.

## 9. Stop/go criteria

Do not claim that graph or hybrid reasoning is superior until:
1. E0 baseline reproduction is complete.
2. E1 uses identical held-out cases and documented denominators.
3. Results are stable across repeated splits or grouped cross-validation.
4. Improvements are reported with uncertainty and alongside coverage/abstention.
5. Any claim of causal correctness is supported by expert-adjudicated evaluation, not row co-occurrence alone.

The experimental basis is established by a reproducible protocol and auditable runs; it is not established merely by implementing an experiment button.
