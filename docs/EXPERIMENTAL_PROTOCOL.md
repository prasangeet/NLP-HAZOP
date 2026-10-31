# Experimental Protocol: NLP for Structured HAZOP Reasoning

**Status:** Protocol draft; not yet a completed evaluation.  
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

## 4. Metrics

For set-valued consequence predictions, compute micro and macro precision, recall, and F1 against the declared target set. Also report:
- Hit@1 and mean reciprocal rank (MRR) for ranked candidates.
- Coverage = queries with at least one proposal / all eligible queries.
- Abstention rate and reason categories.
- Selective risk = error rate among queries answered, reported alongside coverage.
- Mean, median, and percentile latency; LLM token use and estimated cost where available.
- Provenance/evidence traceability rate and unsupported-proposal rate, with operational definitions fixed before evaluation.

State exact-match normalization rules. Preserve raw strings alongside normalized comparison values.

## 5. Experiment sequence

### E0 — Reproduce historical baseline
Run the existing benchmark at seed 42 and verify outputs match the checked-in baseline within expected deterministic tolerances. Investigate discrepancies before adding methods.

**Acceptance:** per-case outputs, configuration, and summary artifacts are generated; tests pass; any mismatch is explained.

### E1 — Controlled strategy comparison
Compare TF-IDF, graph reasoning, prompt-only LLM, and hybrid reasoning on the same split and same case IDs. Run LLM methods only when credentials and model configuration are available. Log failures and cost.

**Acceptance:** a result row per method and dataset, with denominators, coverage, abstentions, latency, metrics, model/version, and output artifact locations.

### E2 — Graph coverage diagnosis
Partition test cases into exact-node match, normalized-node match, supported path, ambiguous path, and no-supported-path groups. This is diagnostic, not a performance improvement by itself.

### E3 — Normalization ablation
Compare exact matching with a canonicalization/normalization variant. Create mappings from training data or a separately reviewed ontology only; do not derive mappings from hidden test targets. Report false merges and per-group metrics.

### E4 — Multi-hop ablation
Compare direct retrieval with graph reasoning under declared path limits (for example, 1, 2, and 4 hops). Report results by evidence-path length and ambiguity class. Only score cases whose reference target and task definition support the comparison.

### E5 — Abstention / selective prediction
Vary confidence or support thresholds on validation data, then evaluate the frozen thresholds on test data. Report coverage–risk curves and abstention reasons. Never tune thresholds on the test set.

### E6 — Cross-domain transfer
Train/index on one domain and evaluate on another without using target-domain test answers for tuning. Report each direction separately; do not treat synthetic-to-synthetic transfer as real-world validation.

### E7 — Expert-adjudicated validation
Have qualified process-safety reviewers independently assess a held-out subset using a written rubric for causal plausibility, contextual support, and safeguard validity. Track disagreements and inter-rater agreement. This is required before making strong claims about causal correctness or safety utility.

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
