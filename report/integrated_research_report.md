# NLP for Structured HAZOP Reasoning
## Integrated Research Report: Literature, System, Dataset Construction, Synthetic Scenarios, and Experiments E1–E7

**Author:** Prasangeet Dongre  
**Roll number:** B23CH1033  
**Department:** Chemical Engineering  
**Advisor:** Dr. Santhosh Kumar Varanasi  
**Repository:** https://github.com/prasangeet/NLP-HAZOP  
**Report status:** Integrated account of repository and frozen experiment evidence reviewed on 7 November 2026.

---

## Abstract

Hazard and Operability Study (HAZOP) worksheets encode process-safety knowledge in structured tables, but their flat presentation does not explicitly preserve intermediate physical states, branching paths, or safeguard attachment points. This project investigates a provenance-aware typed graph representation for HAZOP information and evaluates retrieval and graph reasoning for recovering missing cause-to-consequence worksheet associations.

The work combines a literature review of model-based HAZOP automation, ontology reasoning, NLP extraction, process-safety knowledge graphs, and direct LLM generation with a software prototype comprising typed domain models, graph persistence, sparse retrieval, graph reasoning, report extraction, and a Tauri/React interface. The repository contains three separate data tracks: 40 reconstructed T-301 hexane-storage records, 54 project-generated synthetic T-301 continuations, and 30 synthetic/simulation-derived DWSIM ethanol-fermentation scenarios.

Seven experiments (E1–E7) were conducted on the 40-record reconstructed T-301 track. On the fixed scenario-level holdout, graph reasoning achieved micro-F1 0.7442, compared with 0.3553 for TF–IDF retrieval and 0.5549 for the deterministic graph-plus-TF–IDF union. Follow-on experiments showed that stricter support and provenance-coherence filters sharply reduce coverage, repeated holdouts exhibit split sensitivity, and random removal of training relationships reduces graph-reasoner recall and F1. The available data did not exercise eligible intermediate multi-hop paths.

All E1–E7 targets are reconstructed row-level silver associations: consequences co-listed with causes in a worksheet row. They are not independently adjudicated physical causal relationships. Synthetic and simulation-derived scenarios broaden development coverage but have their own provenance and validation limitations. The results support typed graph reasoning as a promising method for recovering worksheet associations on this dataset; they do not establish causal correctness or operational safety.

**Keywords:** HAZOP; process safety; NLP; knowledge graphs; causal representation; graph reasoning; provenance; synthetic scenarios; DWSIM; missing-relationship recovery.

---

# 1. Introduction

## 1.1 Background and motivation

HAZOP is a systematic qualitative method for identifying hazards and operability problems in process systems. A study divides a process into nodes, examines parameters using guide words, and records deviations, possible causes, consequences, safeguards, and recommendations.

A conventional worksheet row may associate a deviation such as high temperature with a cause such as cooling-pump failure and a consequence such as reactor overpressure. However, the physical sequence may contain intermediate states: reduced cooling, temperature rise, reaction acceleration, vapour generation, and pressure increase. Safeguards may act at different points in this chain. A flat row does not necessarily retain these intermediate states or the attachment point of each safeguard.

The project therefore investigates a typed graph representation of HAZOP information, combined with retrieval and bounded reasoning. The intended role is research assistance: structuring information, surfacing candidate relationships, preserving evidence, and abstaining when the available evidence is insufficient.

## 1.2 Scope and research questions

The broad objective is to investigate whether NLP and LLM-based extraction, typed graph representation, and constrained reasoning can help reconstruct relationships embedded in HAZOP documentation and recover missing worksheet associations.

The implemented and evaluated scope is narrower:

1. Represent structured HAZOP records and typed relationships.
2. Preserve record and relationship provenance.
3. Load and normalize structured CSV datasets.
4. Persist and query graph relationships through in-memory and Neo4j repositories.
5. Retrieve candidates using sparse lexical retrieval and graph neighbourhoods.
6. Apply bounded graph reasoning and hybrid candidate generation.
7. Extract structured information from supported report documents.
8. Evaluate recovery of held-out cause-to-consequence worksheet associations on reconstructed T-301 data.

The repository does not establish a fully validated pipeline from arbitrary unstructured historical reports to expert-approved causal graphs. Scanned-document OCR, complete process-safety validation, and independent expert adjudication remain limitations.

The research questions are:

- **RQ1:** Can HAZOP content be represented as typed relationships while retaining provenance?
- **RQ2:** Can graph reasoning recover held-out row associations more effectively than sparse lexical retrieval?
- **RQ3:** How does recovery change when repeated support is required for complete paths or individual edges?
- **RQ4:** Do provenance completeness and source-coherence constraints improve recovered associations?
- **RQ5:** How sensitive are results to scenario-level splits and removal of training relationships?
- **RQ6:** How can reconstructed, synthetic, and simulation-derived scenarios be used without conflating their evidential status?

---

# 2. Literature Review and Research Positioning

The weekly reports organize prior work into five categories. This section consolidates their findings and retains the important distinctions.

## 2.1 Classical model-based and expert-system HAZOP

Classical automated HAZOP approaches use manually constructed process models, digraphs, qualitative models, and rule bases. They can propagate deviations through process models, but the models themselves must be authored and maintained. The repository reviews work by Vaidhyanathan and Venkatasubramanian, Rahman et al. (ExpHAZOP+), Rodríguez and de la Mata (D-higraphs), and Taylor.

**Implication:** Graph representation itself is not a novel contribution. The relevant questions concern how the graph is acquired, what relationships it represents, what reasoning is performed, and how outputs are validated.

## 2.2 Ontology-based symbolic reasoning: Single et al. (2020)

The Week 02 report studies Single, Schmidt, and Denecke (2020), *Ontology-based computer aid for the automation of HAZOP studies*. The paper describes a system using OWL 2 DL, Owlready2, and the HermiT reasoner. A structured process-unit model and a HAZOP ontology are supplied to a symbolic inference engine.

The ontology represents concepts such as deviation, cause, effect, super-cause, consequence, safeguard, process unit, equipment, substance, process, and circumstances. It supports multi-step chains, branching consequence paths, and multiple causes converging on one effect. The system generates extended HAZOP worksheets and is demonstrated on a hexane storage tank case.

The key boundary recorded in the weekly report is that the process-unit model and ontology are already structured inputs. The system automates reasoning over that knowledge; it does not derive the process model and ontology from arbitrary unstructured historical HAZOP report text.

**Implication:** The project must not claim that prior work has not performed multi-step causal HAZOP reasoning. The narrower question is how relationships are acquired and recovered from available HAZOP records and reports while preserving provenance and allowing abstention.

## 2.3 NLP and statistical text mining

The literature review groups work by Feng et al., Wang et al., Wang and Gu, and Ekramipooya et al. under NLP and statistical text mining. The reviewed tasks include entity extraction, deviation-category prediction, text classification, and recommendation clustering.

These tasks can extract or classify information, but extraction or label prediction is not equivalent to reconstructing a directed causal graph or reasoning over multi-step dependencies.

**Implication:** Extraction, representation, causal reasoning, missing-link inference, and domain validation must be evaluated separately.

## 2.4 Process-safety knowledge graphs and HAZOP completion

The repository reviews process-safety knowledge graph work, including Mao et al. and Li and Zhao’s HAZOPCT. The weekly notes identify HAZOPCT as close prior work: it constructs a HAZOP knowledge graph using a predefined ontology and applies graph-attention-based reasoning for HAZOP completion, including ablation and interpretability analyses.

**Implication:** Knowledge-graph-based HAZOP completion is not unexplored. A defensible project position must specify the input material, relationship provenance, graph construction, and exact recovery task.

## 2.5 Direct LLM worksheet generation

The repository discusses Lee et al. (2026), which studies direct LLM-based HAZOP generation from process diagrams and prompts. The weekly notes record a gap between textual similarity and expert-assessed scenario validity, with reported validity of 19–37% and concerns about safeguard recommendations.

**Implication:** Fluent worksheet generation does not guarantee process-safety validity. Candidate relationships need evidence, structural constraints, and domain review.

## 2.6 Positioning of this project

The implementation combines typed domain representation, provenance tracking, sparse lexical retrieval, graph-neighbourhood retrieval, bounded graph reasoning, hybrid candidate generation, abstention, and structured report extraction.

The most accurate description is a **provenance-aware structured HAZOP retrieval and reasoning prototype**. Its benchmark demonstrates recovery of row-derived associations on a small reconstructed dataset. It has not yet demonstrated expert-validated causal reconstruction across arbitrary industrial reports.

---

# 3. System Design and Implementation

## 3.1 Architecture

The backend follows a layered architecture separating domain concepts from application orchestration and infrastructure.

| Layer | Responsibility | Examples |
|---|---|---|
| Domain | Typed HAZOP entities, relations, ontology constraints, reasoning types | HAZOP records, causal relations, entity types |
| Application | Use cases, retrieval and reasoning strategies | Graph reasoner, TF–IDF retriever, hybrid reasoner |
| Infrastructure | Parsing, persistence, external model clients | CSV loader, Neo4j and in-memory repositories, document extraction |
| API | HTTP endpoints and request/response models | FastAPI dataset and report-extraction routes |
| Desktop frontend | User-facing workflows | Tauri and React interface |

## 3.2 Typed HAZOP representation

The domain model includes process nodes, equipment, parameters, guide words, deviations, causes, effects, consequences, safeguards, and recommendations.

Relationships are directed and typed. The vocabulary includes:

- `causes_deviation`
- `leads_to`
- `leads_to_consequence`
- `co_occurs_with_consequence`
- `co_listed_with_consequence`
- `mitigated_by`
- `recommends_action`
- `has_parameter`
- `located_in`

The ontology constrains valid subject and object entity types. Importantly, `co_listed_with_consequence` is a worksheet-association predicate used for benchmark output; it is not a declaration that the cause physically produces the consequence.

## 3.3 Provenance and validation metadata

Records and relationships can retain dataset identifiers, record type, relationship origin, evidence text, source identifier and location, confidence, expert-verification status, validation status, and endpoint entity types.

The model distinguishes source-verified, reconstructed, synthetic, expert-validated, and unspecified records. Relationship origins distinguish source-derived, reconstructed, synthetic, retrieval-inferred, graph-inferred, model-inferred, and LLM-extracted information.

Record-level validation does not automatically validate every relationship. The loader requires explicit relationship-level expert validation before a relationship can be marked expert-verified.

## 3.4 CSV loading and ambiguity handling

The CSV loader normalizes column aliases and parses list-valued fields. It constructs typed relationships from fields co-listed in one worksheet row.

An important ambiguity rule is implemented:

- If exactly one intermediate effect is listed, effect-to-consequence row associations can be constructed for the listed consequences.
- If multiple intermediate effects are present, the loader does not pair effects and consequences by position or create a Cartesian product. It records a deviation-to-consequence co-occurrence relation because the source row does not specify which effect leads to which consequence.

This avoids manufacturing unsupported causal pairings from a flat row.

## 3.5 Retrieval and reasoning

The evaluated methods include:

1. **TF–IDF retrieval:** Sparse lexical similarity over structured records.
2. **Graph reasoning:** Typed graph traversal and evidence-based candidate recovery.
3. **Graph + TF–IDF union:** Deterministic union of graph and lexical candidates; not a learned fusion or reranking model.
4. **Prompt-only LLM:** Direct prompting without the same graph evidence structure.
5. **Hybrid graph/vector/LLM:** Combines graph and lexical retrieval with an LLM reasoning stage.

The graph-neighbourhood retriever excludes the target scenario to reduce scenario-level leakage. Graph reasoning returns evidence-bearing proposals or abstains when no eligible path is established. Inferred path output retains provenance and must not be treated as independently verified direct causality.

## 3.6 Report extraction and persistence

The repository includes text extraction for supported PDF, TXT, Markdown, and LOG documents, followed by structured LLM extraction. The extraction instructions require evidence quotations and source locations and prohibit inventing missing causal links. Extracted relationships retain LLM-extracted provenance and remain unreviewed until validated.

Text-layer PDF extraction does not imply OCR for scanned or image-only PDFs. The backend supports in-memory and Neo4j graph persistence. The frontend is a Tauri desktop application built with React. The repository evidence does not establish that every desktop workflow has passed complete manual end-to-end validation.

---

# 4. Dataset Construction and Provenance

The repository contains three separate dataset tracks. They must remain separate in reporting, splitting, and interpretation.

## 4.1 Dataset overview

| Track | Canonical file | Records | Provenance | Role |
|---|---|---:|---|---|
| T-301 reconstructed | `data/raw/t301/T301_HAZOP_CCPS2001_style.csv`, filtered by record type | 40 | Reconstructed from the reported T-301 hexane-storage case in Single et al. (2020); not verbatim source rows | Primary E1–E7 evaluation |
| T-301 synthetic continuation | Same T-301 CSV, filtered by record type | 54 | Project-generated synthetic continuations | Development and separate exploratory track |
| DWSIM synthetic | `data/raw/dwsim/synthetic_hazop_D001_D030.csv` | 30 | Synthetic HAZOP scenarios informed by DWSIM simulations and engineering hypotheses | Simulation-informed development and separate exploratory track |

The DWSIM worksheet-format CSV is an alternate representation of D001–D030, not an independent second sample.

## 4.2 T-301 reconstructed track

The 40 reconstructed records describe scenario chains associated with the hexane storage case. They are structured reconstructions, not verbatim transcriptions of the original expert worksheet.

The frozen dataset SHA-256 is:

`1c8713c2b6fbc9cdf2e22e64a307fdb64f073158a78679bdc23ad803a7d2725e`

This is the only track used for E1–E7. Its benchmark targets are row-level silver associations, not independently adjudicated causal relations.

## 4.3 T-301 synthetic continuation: 54 records

The same T-301 CSV contains 54 records marked `SYNTHETIC_CONTINUATION`. They extend the T-301 hexane-storage context with additional deviations, failure modes, effects, consequences, and candidate recommendations.

Examples include high liquid level during transfer, erroneous level indication, blocked outlet, external leakage from flanges or drain valves, tank-wall leakage, instrument-connection leakage, vent-path discharge, maintenance errors, and external impact.

The records explicitly state that they are not established by the source case and must not be attributed to CCPS (2001) or Single et al. (2020). They are labelled synthetic continuation and engineering review required.

**Generation-method evidence boundary:** The repository documents these 54 records as project-generated synthetic continuations and contains their scenario descriptions and provenance notes. The reviewed repository files do not provide a separate deterministic scenario-generation script, prompt log, model version, random seed, or expert-approval record for these rows. Accordingly, this report describes them as project-generated synthetic scenario continuations based on the T-301 context and plausible failure/deviation patterns. It does not claim an automated, reproducible generation procedure that is not documented.

These records are useful for testing schema handling and exploring additional scenario descriptions. They are not source-verified data or expert ground truth.

## 4.4 DWSIM ethanol/fermentation track: 30 records

The canonical file is `data/raw/dwsim/synthetic_hazop_D001_D030.csv`. It covers four process nodes:

- N1: heater/feed section, including H-1 and feed conditions.
- N2: reactor section, including R-1 and fermentation conditions.
- N3: cooler section, including CL-1 and cooling/throughput conditions.
- N4: separator section, including SEP-1 and feed/separation conditions.

Parameters include temperature, flow, pressure, composition, conversion, heat removal, cooling duty, and separation efficiency. Guide words include MORE, LESS, NO, and OTHER THAN.

### 4.4.1 Simulation-informed scenario construction

The DWSIM track differs from the T-301 synthetic continuation because its effect descriptions are tied, where possible, to process-simulation perturbations. The CSV records manipulated variables, observed or inferred responses, and evidence/limitation notes.

| Scenarios | Perturbation or case | Recorded response | Qualification |
|---|---|---|---|
| D001 | Heater outlet temperature above baseline | Temperature and downstream product response changed | Consequences and safeguards remain synthetic hypotheses |
| D003 | No heater duty / heater failure | Feed temperature remained near inlet temperature | Downstream consequence requires review |
| D004–D005 | Feed flow increased/decreased | Whole-flowsheet stream flows changed by about 10% | Not isolated to one local item |
| D006–D007 | Feed pressure increased/decreased | Vapor/liquid and product flows shifted slightly | Operating limits unavailable |
| D008–D009 | Reactor outlet temperature changed | Simulated vapor flow/composition changed | Causal pathway not validated |
| D014 | Feed ethanol fraction changed | Reactor outlet and product compositions changed | Molar flow also changed |
| D015 | Fermentation conversion reduced | Conversion setting and energy response changed | Reaction model requires review |
| D016 | Reactor heat removal reduced | Model showed negligible response | Reliable thermal consequence could not be inferred |
| D017–D018 | Cooler outlet temperature changed | Beer flow, ethanol fraction, and vent composition shifted | Product-quality implications need review |
| D021 | Cooling duty reduced by 10% | Outlet temperature rose; beer mass flow fell about 11.07%; composition changed | Safety consequence needs validation |
| D022 | No cooling duty | Zero beer-stream flow with anomalous reported properties | Internally questionable converged result |
| D023–D026 | Separator pressure/temperature changed | Connected stream outputs did not show validated response | Model revision required |
| D029 | Reduced separator efficiency | No valid writable perturbation was available | Hypothetical; not simulated |
| D030 | Feed ethanol fraction increased by 0.01 and water decreased by 0.01 | Beer mass flow, ethanol mole fraction, and CO2 vent mass flow changed | Fixed mass-flow manipulation also changed molar flow |

The DWSIM records are therefore not simply invented cause/consequence strings: some effects are tied to simulation output. However, a simulation response is not automatically a validated hazard consequence. Some manipulations affected the whole flowsheet, some model responses were questionable, and some consequences remained hypothetical.

### 4.4.2 Review-status distribution

The dataset README records:

| Status | Records |
|---|---:|
| Review required | 16 |
| Scope caveat | 6 |
| NEEDS_MODEL_REVISION | 6 |
| Review reaction model | 1 |
| NEEDS_REVIEW | 1 |
| **Total** | **30** |

These labels identify evidence limitations and review needs; none means expert validation has been completed.

### 4.4.3 Alternate worksheet export

`HAZOP_DWSIM30_worksheet_format.csv` is a worksheet-style export of the same D001–D030 scenarios. It is not an independent sample and must not be counted as an additional 30 records.

## 4.5 Appropriate use of synthetic scenarios

Synthetic tracks are useful for testing CSV loading, provenance mapping, ontology constraints, graph persistence, retrieval, and application workflows. They can expose missing schema cases and model limitations.

They do not establish expert-approved HAZOP completeness, physical causality of every listed relationship, safeguard adequacy, generalization to arbitrary industrial plants, or independent validation merely because an effect was generated by DWSIM.

For future scenario batches, preserve a generation manifest containing authoring method, model/provider and version if applicable, prompt/template version, random seed if applicable, date, process-model version, and source files. Separate observed simulation output from engineering interpretation and generated narrative. Record manipulated and baseline values, perturbation magnitude, convergence status, affected streams/equipment, and output changes. Mark each cause, effect, consequence, and safeguard as simulated, inferred, hypothesized, or expert-reviewed.

---

# 5. Benchmark Design

## 5.1 Primary evaluation protocol

E1–E7 use only the 40-record reconstructed T-301 track. The synthetic continuation and DWSIM records are not pooled into these experiments.

The fixed split uses:

- Protocol version 1.3.
- Seed 42.
- Scenario-level 80/20 split.
- 32 training scenarios and 8 held-out scenarios.
- 19 test queries.
- Candidate limit top-10.
- Maximum graph depth 4.

All rows belonging to one scenario remain in one partition. For each held-out cause, the silver target is the set of consequences co-listed in that same worksheet row. The held-out target labels and graph edges are removed from the retrieval repository.

## 5.2 Metrics

- **Precision:** Fraction of proposed associations matching the silver target set.
- **Recall:** Fraction of silver target associations recovered.
- **F1:** Harmonic mean of precision and recall.
- **Coverage:** Fraction of queries receiving at least one proposal.
- **Abstention:** Fraction of queries receiving no proposal.
- **Selective risk:** Error rate among answered queries.
- **Latency:** Runtime under the recorded implementation and environment.

Coverage must be read with precision and recall. A method can return candidates for every query and still be inaccurate. A conservative method may abstain to avoid unsupported proposals.

## 5.3 Evaluation boundary

The benchmark measures agreement with reconstructed row-level silver associations. It does not measure expert-adjudicated causal truth. Scenario-level splitting reduces direct row leakage, but does not create independent physical evidence or eliminate shared assumptions across reconstructed scenarios.

---

# 6. Experimental Results E1–E7

## 6.1 E1 — Controlled strategy comparison

E1 compares sparse TF–IDF retrieval, graph reasoning, deterministic graph-plus-TF–IDF union, prompt-only LLM reasoning, and hybrid graph/vector/LLM reasoning. LLM conditions were capped at 15 of the 19 test queries; their denominators differ from deterministic methods.

### Full-test results

| Method | Evaluated / 19 | Precision | Recall | Micro-F1 | Coverage | Abstention | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prompt-only LLM | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0667 | 0.9333 | 16,802.4044 ms |
| TF–IDF retrieval | 19 | 0.2784 | 0.4909 | 0.3553 | 0.8947 | 0.1053 | 0.5075 ms |
| Graph reasoning | 19 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 | 1.3385 ms |
| Graph + TF–IDF union | 19 | 0.4068 | 0.8727 | 0.5549 | 0.8947 | 0.1053 | 1.8460 ms |
| Hybrid graph/vector/LLM | 15 | 0.3704 | 0.6383 | 0.4688 | 0.8667 | 0.1333 | 38,415.4310 ms |

### Matched 15-query comparison

| Method | Precision | Recall | Micro-F1 | Coverage |
|---|---:|---:|---:|---:|
| Prompt-only LLM | 0.0000 | 0.0000 | 0.0000 | 0.0667 |
| TF–IDF retrieval | 0.3462 | 0.5745 | 0.4320 | 0.8667 |
| Graph reasoning | 0.6452 | 0.8511 | 0.7339 | 0.8000 |
| Graph + TF–IDF union | 0.4396 | 0.8511 | 0.5797 | 0.8667 |
| Hybrid graph/vector/LLM | 0.3704 | 0.6383 | 0.4688 | 0.8667 |

**Finding:** Graph reasoning is the strongest deterministic method on this split. The union increases coverage but adds false positives and lowers F1. The hybrid method does not outperform graph reasoning on the matched subset and is much slower in this run. Prompt-only LLM almost always abstains. Full-test rows must not be compared as if all methods evaluated the same number of queries.

## 6.2 E2 — Global complete-path support

E2 requires the same complete typed Cause → Deviation → Consequence path to be supported by a minimum number of distinct eligible training scenarios, excluding the target scenario.

| Minimum supporting scenarios | Precision | Recall | F1 | Coverage | Abstention |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |
| 2 | 0.6000 | 0.1091 | 0.1846 | 0.3158 | 0.6842 |
| 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

**Finding:** Complete-path recurrence is sparse. Requiring support from two scenarios sharply reduces recall and coverage; thresholds of three or more eliminate all proposals. Scenario recurrence is not independent physical confirmation.

## 6.3 E3 — Independent typed-edge support

E3 counts support for Cause → Deviation and Deviation → Consequence edges independently. The two edges may be supported by different training scenarios.

| Condition | Precision | Recall | F1 | Coverage | Abstention |
|---|---:|---:|---:|---:|---:|
| Both edge thresholds = 1 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |
| Cause edge ≥ 2 | 0.5938 | 0.3455 | 0.4368 | 0.3158 | 0.6842 |
| Consequence edge ≥ 2 | 0.6094 | 0.7091 | 0.6555 | 0.8421 | 0.1579 |
| Both edges ≥ 2 | 0.6000 | 0.3273 | 0.4235 | 0.3158 | 0.6842 |

**Finding:** Cause-to-deviation support is the stronger bottleneck. Raising its threshold sharply reduces recall and coverage. Raising only the consequence-edge threshold has a smaller effect. Independent edge support does not provide a sufficient precision improvement to compensate for lost coverage.

## 6.4 E4 — Multi-hop reasoning

| Path mode | Candidate paths | Precision | Recall | F1 | Coverage | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 265 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |
| Intermediate | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Both | 265 | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |

**Finding:** No eligible intermediate multi-hop paths were available under the dataset and constraints. The combined condition therefore reproduces the direct-path result. This is an evidence-availability limitation, not proof that multi-hop reasoning is intrinsically ineffective.

## 6.5 E5 — Provenance completeness and source coherence

### E5-1: Metadata-presence thresholds

Metadata-presence thresholds of 0, 0.5, 0.75, and 1.0 produced identical results:

| Threshold | Precision | Selective risk | F1 | Coverage | Abstention | Unsupported | Traceable | Accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.6486 | 0.3514 | 0.7442 | 0.8421 | 0.1579 | 0 | 1.0000 | 74 |
| 0.50 | 0.6486 | 0.3514 | 0.7442 | 0.8421 | 0.1579 | 0 | 1.0000 | 74 |
| 0.75 | 0.6486 | 0.3514 | 0.7442 | 0.8421 | 0.1579 | 0 | 1.0000 | 74 |
| 1.00 | 0.6486 | 0.3514 | 0.7442 | 0.8421 | 0.1579 | 0 | 1.0000 | 74 |

The accepted set contains 48 true positives and 26 false positives, with 7 false negatives.

**Finding:** All proposals were traceable under the tested accounting, but metadata presence did not discriminate correct from incorrect proposals. Provenance completeness is not causal validation.

### E5-2: Source-coherence filters

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

**Finding:** Both strict filters behave identically in this run. They reject 62 of 74 proposals, retaining 6 true positives and 6 false positives. Precision falls and recall collapses. Documentary coherence supports traceability but does not establish causal correctness; a strict same-source constraint can also remove useful cross-scenario evidence.

## 6.6 E6 — Scenario-split robustness

E6 repeats the scenario-level 80/20 holdout for ten seeds, 42–51. Each run holds out 8 of 40 scenarios. Query counts vary from 17 to 27. Across the runs, 36 unique scenarios appear in a test partition and 4 never appear as test scenarios.

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Micro precision | 0.5765 | 0.0412 | 0.5294 | 0.6486 |
| Micro recall | 0.8615 | 0.1034 | 0.6383 | 1.0000 |
| Micro F1 | 0.6881 | 0.0479 | 0.5825 | 0.7442 |
| Macro F1 | 0.6266 | 0.0973 | 0.3722 | 0.7057 |
| Coverage | 0.8746 | 0.1283 | 0.5500 | 1.0000 |
| Abstention | 0.1254 | 0.1283 | 0.0000 | 0.4500 |

**Finding:** Results vary with the scenario partition. These are repeated random holdouts, not ten-fold cross-validation: training partitions overlap, and runs are not independent. The mean and standard deviation are descriptive, not an independent-sample confidence interval.

## 6.7 E7 — Controlled relationship-evidence masking

E7 holds the seed-42 split fixed: 32 training scenarios, 8 test scenarios, and 19 queries. It randomly masks 10%, 20%, 30%, and 40% of 181 typed training relationships. Ten masking seeds (42–51) are used at each rate, with masks nested within each seed. Test labels are unchanged and LLM methods are disabled. This produces 40 masking conditions.

### Graph reasoning

| Masked relationships | Mean precision | Mean recall | Mean F1 | Mean coverage | Mean abstention |
|---:|---:|---:|---:|---:|---:|
| 0% baseline | 0.6486 | 0.8727 | 0.7442 | 0.8421 | 0.1579 |
| 10% | 0.6454 | 0.8109 | 0.7182 | 0.8000 | 0.2000 |
| 20% | 0.6439 | 0.7382 | 0.6872 | 0.7316 | 0.2684 |
| 30% | 0.6396 | 0.6582 | 0.6469 | 0.6842 | 0.3158 |
| 40% | 0.6476 | 0.5727 | 0.6030 | 0.6158 | 0.3842 |

### Graph + TF–IDF union

| Masked relationships | Mean precision | Mean recall | Mean F1 | Macro-F1 | Mean coverage | Mean abstention |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.4077 | 0.8145 | 0.5432 | 0.4892 | 0.8737 | 0.1263 |
| 20% | 0.4124 | 0.7455 | 0.5305 | 0.4559 | 0.8526 | 0.1474 |
| 30% | 0.4190 | 0.6673 | 0.5139 | 0.4248 | 0.8263 | 0.1737 |
| 40% | 0.4299 | 0.5855 | 0.4936 | 0.3897 | 0.7737 | 0.2263 |

### TF–IDF retrieval

| Masked relationships | Mean precision | Mean recall | Mean F1 | Macro-F1 | Mean coverage |
|---:|---:|---:|---:|---:|---:|
| 10% | 0.2713 | 0.4182 | 0.3288 | 0.2787 | 0.8684 |
| 20% | 0.2770 | 0.3655 | 0.3143 | 0.2581 | 0.8368 |
| 30% | 0.2663 | 0.2945 | 0.2775 | 0.2141 | 0.8053 |
| 40% | 0.2570 | 0.2291 | 0.2402 | 0.1792 | 0.7526 |

**Finding:** As masking increases, graph-reasoner recall, F1, and coverage decline while precision remains comparatively stable. Evidence loss primarily reduces recoverable associations and increases abstention. The union retains higher coverage but lower F1 than graph reasoning. TF–IDF is weakest in this experiment.

The masking procedure is controlled deletion of loader-generated typed relationships, not a model of naturally occurring missingness in industrial reports. The 0% row is the E1 baseline, not an additional randomized masking condition.

---

# 7. Cross-Experiment Discussion

1. **Graph reasoning leads on the fixed reconstructed split.** E1 shows micro-F1 0.7442 versus 0.3553 for TF–IDF. This supports typed graph context for the particular row-derived associations represented in the T-301 benchmark, not physical causal validity.
2. **Coverage and precision trade off.** The graph-plus-TF–IDF union increases coverage but introduces false positives and reduces F1. It is a coverage-oriented candidate generator, not a superior accuracy method.
3. **Repeated support is sparse.** E2 and E3 show that recurrence thresholds sharply reduce recall and coverage, especially for cause-to-deviation edges.
4. **Provenance is necessary but insufficient.** E5 shows that traceability and metadata completeness do not identify causal truth. Strict same-source filters can over-filter useful cross-scenario evidence.
5. **Multi-hop reasoning remains under-tested.** E4 found no eligible intermediate paths. A meaningful multi-hop claim requires a dataset with explicit intermediate events and adjudicated paths.
6. **Scenario composition matters.** E6 shows split sensitivity; repeated holdouts are not independent cross-validation folds.
7. **Evidence loss reduces recoverability.** E7 shows that random relationship deletion reduces recall, F1, and coverage while increasing abstention.

---

# 8. Earlier Three-Track Baseline (30 October 2026)

The repository also contains an exploratory baseline report dated 30 October 2026. It is a separate earlier implementation/evaluation snapshot and must not be merged with E1–E7.

## 8.1 Protocol and split

The baseline used a scenario-level 80/20 split with seed 42, top-10 candidate limit, graph search cap of 4 hops, one query per held-out cause, and the consequences co-listed in the held-out row as the silver target. LLM methods were not run in this CI baseline.

| Track | Records | Train scenarios | Test scenarios | Test queries |
|---|---:|---:|---:|---:|
| T-301 reconstructed | 40 | 32 | 8 | 19 |
| T-301 synthetic continuation | 54 | 43 | 11 | 11 |
| DWSIM synthetic | 30 | 24 | 6 | 15 |

## 8.2 TF–IDF results

| Metric | T-301 reconstructed | T-301 synthetic | DWSIM synthetic |
|---|---:|---:|---:|
| Micro precision | 0.2680 | 0.0000 | 0.0900 |
| Micro recall | 0.4727 | 0.0000 | 0.3462 |
| Micro F1 | 0.3421 | 0.0000 | 0.1429 |
| Macro F1 | 0.2968 | 0.0000 | 0.1343 |
| Hit@1 | 0.1579 | 0.0000 | 0.2667 |
| MRR | 0.3456 | 0.0000 | 0.3278 |
| Coverage | 0.8947 | 1.0000 | 0.9333 |
| Abstention | 0.1053 | 0.0000 | 0.0667 |
| Mean candidates/query | 5.11 | 7.09 | 6.67 |
| Mean query latency | 0.151 ms | 0.371 ms | 0.131 ms |
| Index build time | 0.754 ms | 1.330 ms | 0.551 ms |

All returned TF–IDF candidates carried retrieval-inferred provenance. The baseline report records a 0% source-or-expert origin rate for these candidates; they are row analogies, not direct source-verified edges.

## 8.3 Graph reasoning results

Graph reasoning abstained on all queries in this earlier baseline:

| Track | Precision | Recall | F1 | Coverage | Abstention |
|---|---:|---:|---:|---:|---:|
| T-301 reconstructed | 0 | 0 | 0 | 0 | 1.0000 |
| T-301 synthetic | 0 | 0 | 0 | 0 | 1.0000 |
| DWSIM synthetic | 0 | 0 | 0 | 0 | 1.0000 |

The baseline report attributes reconstructed-track abstentions to ambiguous competing terminal paths and synthetic-track abstentions to insufficient supported training paths. This is a limitation observed at that earlier repository state, not evidence that graph reasoning is intrinsically ineffective.

## 8.4 Interpretation

TF–IDF returned candidates at high coverage but did not recover any silver targets on the T-301 synthetic continuation track and achieved low F1 on DWSIM synthetic. The DWSIM targets include simulation/inference-derived effects and unreviewed engineering hypotheses; matching them does not establish physical validity.

The 30 October baseline and later E1–E7 results belong to different implementation/evaluation snapshots. They should be presented chronologically, not combined as one run.

---

# 9. Synthetic Scenario Development: Analysis and Recommendations

## 9.1 Why synthetic scenarios were added

The reconstructed T-301 track is small. Synthetic records broaden the vocabulary of equipment, parameters, guide words, deviations, failure modes, effects, consequences, safeguards, and recommendations. They also exercise schema handling and application workflows across additional records and a second process context.

## 9.2 Two distinct synthetic-data processes

The project has two materially different synthetic tracks:

- **T-301 synthetic continuation:** Project-generated scenario continuations in the same hexane-storage context. The repository contains scenario text and explicit provenance notes, but no reproducible generation script, prompt trace, model metadata, or seed was found in the reviewed paths. These should be described as project-generated synthetic hypotheses.
- **DWSIM synthetic track:** Thirty scenario records combining HAZOP-style hypotheses with DWSIM perturbation observations where available. The evidence column records captured responses, whole-flowsheet effects, questionable model responses, or scenarios that were not successfully simulated.

They must not be collapsed into one generic synthetic-generation method.

## 9.3 Evidence categories for DWSIM

For future analysis, DWSIM records should be interpreted using explicit evidence categories:

- **Simulation response captured:** The manipulated variable produced an observed model response.
- **Scope caveat:** The manipulation affected a larger portion of the flowsheet than the named local node.
- **Model revision required:** The current model/interface did not provide a valid response to the intended perturbation.
- **Review required:** A response exists, but the HAZOP consequence and safeguard interpretation still requires engineering review.
- **Hypothesis only:** A cause, consequence, or safeguard is proposed but not demonstrated by simulation.

The current CSV partially encodes these distinctions through validation status and free-text evidence/limitation notes. Future versions should make them explicit structured fields.

## 9.4 Risks of synthetic evaluation

Synthetic evaluation can be misleading when the same assumptions are used to generate and evaluate scenarios. A synthetic cause and consequence may have been written together by the same author; model-generated scenarios may encode the expected answer pattern; a simulation perturbation may not represent a realistic equipment failure; a converged model may still produce questionable outputs; and candidate safeguards may not exist in an actual plant.

Synthetic records should therefore remain separate from reconstructed and expert-validated data. They are useful for software development and hypothesis exploration, but not a substitute for independent process-safety review.

---

# 10. Implementation Status, CI Evidence, and Limitations

## 10.1 Capabilities present in the repository

The reviewed repository contains typed HAZOP domain records and relationships, ontology endpoint constraints, provenance and validation metadata, CSV loading, in-memory and Neo4j graph persistence, TF–IDF retrieval, graph-neighbourhood retrieval, graph and hybrid reasoning, benchmark scripts, report text extraction, structured LLM extraction routes, a FastAPI backend, and a Tauri/React frontend.

## 10.2 CI status snapshot

The reviewed CI snapshot at merge commit `c7596c4e66634f2d438abfbe0b4890e797df1d95` showed:

- Ruff lint failed with 20 diagnostics.
- The dependent unit-test step was skipped in that workflow.
- Neo4j integration testing failed during dataset import, with an API 502 following a Neo4j Query API 400.
- The benchmark job was skipped.
- The snapshot did not establish a complete Tauri build or manual end-to-end workflow.

This is a historical snapshot and may have changed after subsequent commits. It must not be presented as current CI status without rechecking GitHub Actions.

## 10.3 Principal limitations

1. The primary reconstructed benchmark contains only 40 scenarios, 8 held-out scenarios, and 19 fixed-split queries.
2. Targets are reconstructed row-level silver associations, not expert-adjudicated causal links.
3. No independent relationship-level causal gold set is available.
4. E4 did not exercise eligible intermediate paths.
5. The T-301 synthetic continuation has no repository-tracked reproducible generation manifest in the reviewed files.
6. DWSIM scenarios contain model, scope, and engineering-review caveats.
7. E6 is repeated random holdout, not cross-validation.
8. E7 uses uniform random deletion, not naturally occurring documentation missingness.
9. LLM evaluation was capped at 15 queries in E1; provider cost/token accounting was not captured.
10. Text-layer PDF extraction does not provide OCR for scanned documents.
11. The deterministic safety validator is structural validation, not process simulation or expert review.
12. The reviewed CI snapshot does not establish full desktop end-to-end validation.

## 10.4 Safety boundary

The prototype is a research tool for structuring, retrieving, and reviewing HAZOP information. Recovered links, generated scenarios, and candidate safeguards must not be treated as verified process-safety facts or operational recommendations. They do not replace qualified HAZOP facilitation, process engineering review, or formal safety validation.

---

# 11. Conclusions and Future Work

The literature review shows that model-based and ontology-based causal HAZOP reasoning, process-safety knowledge graphs, NLP extraction, and direct LLM generation all have substantial prior work. The project contribution is therefore not simply the use of graphs for HAZOP. It is the investigation of a provenance-aware workflow combining typed relationship representation, retrieval, bounded graph reasoning, abstention, and missing-association evaluation.

On the reconstructed T-301 benchmark, graph reasoning is the strongest tested deterministic method on the fixed seed-42 split, achieving micro-F1 0.7442. The graph-plus-TF–IDF union increases coverage but reduces precision and F1. The tested LLM conditions do not outperform graph reasoning on the matched subset and incur much higher latency in the recorded run.

E2 and E3 show that repeated support requirements sharply reduce recall and coverage because complete paths and cause edges are sparse. E4 does not exercise intermediate paths. E5 shows that provenance completeness is not a proxy for correctness and that strict source coherence can over-filter. E6 demonstrates split sensitivity. E7 shows that random relationship deletion reduces graph-reasoner recall, F1, and coverage.

The synthetic-data work broadens development coverage but has different evidence levels. The 54 T-301 continuations are project-generated scenario hypotheses, not paper-reported records. The 30 DWSIM scenarios combine simulated process responses with engineering hypotheses and model limitations. Neither synthetic track should be treated as expert-validated ground truth or pooled with the reconstructed T-301 E1–E7 results.

The most important next step is independent expert adjudication of a held-out set of relationships and scenarios. A written rubric should cover causal plausibility, context sufficiency, intermediate-event validity, safeguard existence and adequacy, and uncertainty. A larger and more diverse dataset, realistic missingness protocol, explicit intermediate-event annotations, and end-to-end application validation are also needed.

**Final conclusion:** Current evidence supports typed graph reasoning as a promising method for recovering reconstructed HAZOP worksheet associations in the T-301 case. It does not yet establish expert-approved causal inference, physical validity, or operational process-safety utility.

---

# 12. Repository Sources and Reproducibility References

## 12.1 Reports and research notes

- `report/main.tex` — weekly research log master document and report structure.
- `report/weeks/week01.tex` — HAZOP fundamentals, literature taxonomy, capability framework, initial research framing.
- `report/weeks/week02.tex` — detailed analysis of Single et al. (2020).
- `report/weeks/week03.tex` — HAZOPCT analysis, research design, graph representation, initial methodology.
- `report/weeks/week04.tex` — E1–E7 experimental evaluation.
- `report/references.bib` — shared bibliography.
- `RESEARCH_DOCUMENTATION.md` — research thesis, literature taxonomy, system architecture, dataset provenance, and baseline discussion.
- `reports/benchmark-baseline-2026-10-30.md` — earlier three-track exploratory baseline.
- `reports/experimental-synthesis-e1-e7.md` — frozen E1–E7 results and interpretation.

## 12.2 Dataset sources

- `data/README.md` — dataset provenance and separation rules.
- `data/raw/t301/T301_HAZOP_CCPS2001_style.csv` — 40 reconstructed and 54 synthetic continuation records.
- `data/raw/dwsim/synthetic_hazop_D001_D030.csv` — canonical 30-record DWSIM synthetic dataset.
- `data/raw/dwsim/HAZOP_DWSIM30_worksheet_format.csv` — alternate export of D001–D030, not an independent sample.

## 12.3 Implementation and experiment sources

- `server/app/domain/entities/hazop_record.py` — HAZOP record, causal relation, provenance, and validation models.
- `server/app/domain/ontology/schema.py` — typed relationship constraints.
- `server/app/infrastructure/datasets/hazop_csv_loader.py` — CSV normalization and relationship construction.
- `server/app/application/strategies/graph_reasoner.py` — graph reasoning.
- `server/app/application/strategies/hybrid_reasoner.py` — hybrid reasoning.
- `server/app/application/strategies/retrieval/tfidf.py` — sparse lexical retrieval.
- `server/app/application/strategies/retrieval/graph_neighborhood.py` — graph-neighbourhood retrieval.
- `server/app/domain/services/safety_validator.py` — deterministic structural validation.
- `server/app/infrastructure/extraction/document_text_extractor.py` — document text extraction.
- `server/app/infrastructure/extraction/gemini_report_extractor.py` — structured LLM report extraction.
- `server/app/application/use_cases/extract_and_store_report.py` — report extraction orchestration.
- `server/app/api/routes/report_extraction.py` — report extraction API.
- `server/scripts/run_benchmark.py` — benchmark runner.
- `server/scripts/run_experiment_2.py` through `server/scripts/run_experiment_7.py` — E2–E7 runners.
- `server/results/e7/e7_summary_20261106T151612Z.json` — E7 summary artifact.
- `server/results/e7/e7_runs_20261106T151612Z.jsonl` — E7 per-condition run artifact.

## 12.4 Reproducibility notes

The reconstructed T-301 dataset hash is recorded in Section 4. E1–E7 summaries should be treated as frozen results. Future reruns should preserve the exact commit SHA, dataset hash, protocol version, split seed, method configuration, model/provider version where relevant, and timestamp.

The 30 October baseline and later E1–E7 results belong to different implementation/evaluation snapshots. They should remain separately identified in future publications and presentations.
