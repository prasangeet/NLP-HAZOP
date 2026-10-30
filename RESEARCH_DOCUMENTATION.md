# Research Documentation: NLP for Structured HAZOP Reasoning

**Project Title:** Natural Language Processing for Structured HAZOP Reasoning  
**Author:** Prasangeet Dongre (Roll No. B23CH1033)  
**Department:** Department of Chemical Engineering  
**Advisor:** Dr. Santhosh Kumar Varanasi  
**Repository:** `NLP-HAZOP`  
**Date / Version:** October 2026  

---

## Table of Contents

1. [Executive Summary & Core Research Thesis](#1-executive-summary--core-research-thesis)
2. [Process Safety Fundamentals & The HAZOP Problem](#2-process-safety-fundamentals--the-hazop-problem)
   - [2.1 HAZOP Core Concepts](#21-hazop-core-concepts)
   - [2.2 The Tabular Representation Paradox](#22-the-tabular-representation-paradox)
3. [Comprehensive Literature Taxonomy & Prior Art](#3-comprehensive-literature-taxonomy--prior-art)
   - [Category A: Classical Model-Based & Expert Systems](#category-a-classical-model-based--expert-systems)
   - [Category B: Ontology-Based Symbolic Causal Reasoning (Deep Dive: Single et al., 2020)](#category-b-ontology-based-symbolic-causal-reasoning-deep-dive-single-et-al-2020)
   - [Category C: NLP & Statistical Text Mining](#category-c-nlp--statistical-text-mining)
   - [Category D: Process Safety Knowledge Graphs (Deep Dive: HAZOPCT & Mao et al.)](#category-d-process-safety-knowledge-graphs-deep-dive-hazopct--mao-et-al)
   - [Category E: Direct LLM Worksheet Generation (Deep Dive: Lee et al., 2026)](#category-e-direct-llm-worksheet-generation-deep-dive-lee-et-al-2026)
4. [The 5-Level Capability Framework & Identified Research Gap](#4-the-5-level-capability-framework--identified-research-gap)
5. [The Working Research Hypothesis & Proposed Pipeline](#5-the-working-research-hypothesis--proposed-pipeline)
   - [5.1 Formal Graph Representation Schema](#51-formal-graph-representation-schema)
   - [5.2 Causal Proposition Formulation](#52-causal-proposition-formulation)
6. [Software Architecture & Technical Implementation (`server/`)](#6-software-architecture--technical-implementation-server)
   - [6.1 Clean Architecture / Domain-Driven Design (DDD)](#61-clean-architecture--domain-driven-design-ddd)
   - [6.2 Domain Models & Strict Provenance Tracking](#62-domain-models--strict-provenance-tracking)
   - [6.3 Reasoning & Retrieval Strategies](#63-reasoning--retrieval-strategies)
   - [6.4 FastAPI Service & CSV Preview Pipeline](#64-fastapi-service--csv-preview-pipeline)
7. [Datasets, Provenance Rules & Epistemic Hygiene](#7-datasets-provenance-rules--epistemic-hygiene)
   - [7.1 T-301 Hexane Storage Tank Track](#71-t-301-hexane-storage-tank-track)
   - [7.2 DWSIM Ethanol / Fermentation Plant Track](#72-dwsim-ethanol--fermentation-plant-track)
   - [7.3 The Silver Label Caveat](#73-the-silver-label-caveat)
8. [Leakage-Aware Benchmark & Baseline Findings](#8-leakage-aware-benchmark--baseline-findings)
   - [8.1 Benchmark Design](#81-benchmark-design)
   - [8.2 Baseline Run Results (30 October 2026)](#82-baseline-run-results-30-october-2026)
   - [8.3 Analysis of Results & Abstention Dynamics](#83-analysis-of-results--abstention-dynamics)
9. [Open Research Questions & Experimental Roadmap](#9-open-research-questions--experimental-roadmap)

---

## 1. Executive Summary & Core Research Thesis

Hazard and Operability (HAZOP) analysis is the premier systematic qualitative safety evaluation technique employed across the chemical process industries to identify potential operational hazards, deviations, and operability malfunctions. Historically, industrial HAZOP studies are recorded in tabular worksheets that map deviations to causes, consequences, and safeguards in flat rows.

Recent artificial intelligence literature in process safety has attempted to automate or assist HAZOP analysis using Natural Language Processing (NLP), Knowledge Graphs (KGs), and Large Language Models (LLMs). However, a critical disconnect exists in the current research landscape:
1. **Model-based and ontology reasoners** (e.g., Single et al., 2020) perform formal, multi-step causal reasoning, but require exhaustive manual ontological engineering before reasoning can take place.
2. **Text mining and NLP classifiers** (e.g., Ekramipooya et al., 2023) extract labels from report text, but do not construct or reason over an underlying causal graph.
3. **End-to-end multimodal LLMs** (e.g., Lee et al., 2026) attempt to generate worksheets directly from process flow diagrams or text prompts, achieving superficial textual similarity but suffering from dangerously low scenario validity (19% to 37%) and hallucinated safeguards.

### The Central Thesis
> **Can natural-language processing and large language models reliably reconstruct the underlying causal structures embedded in historical HAZOP reports, formalizing them into a structured graph that supports multi-hop causal reasoning, missing-relationship inference, and process-safety constraint validation?**

This project investigates how to transition from **flat document extraction** to **graph-based multi-hop causal reasoning**, enforcing strict data provenance, abstention under ambiguity, and process-safety constraints.

---

## 2. Process Safety Fundamentals & The HAZOP Problem

### 2.1 HAZOP Core Concepts
A traditional HAZOP follows IEC 61882 standards. An interdisciplinary team systematically studies process equipment using standardized concepts:
- **Node:** A discrete process section or equipment boundary (e.g., *Hexane Storage Tank T-301*, *Reboiler Loop*).
- **Parameter:** A physical process variable (e.g., *Temperature*, *Pressure*, *Flow*, *Level*, *Composition*).
- **Guide Word:** A standardized qualitative modifier (e.g., `NO`, `MORE`, `LESS`, `REVERSE`, `PART OF`, `AS WELL AS`).
- **Deviation:** The pairing of a Guide Word with a Parameter (e.g., `MORE` + `Temperature` $\rightarrow$ *High Temperature*).
- **Cause:** An initiating physical, mechanical, or human event that provokes the deviation (e.g., *Cooling pump mechanical seal failure*).
- **Consequence:** The hazardous downstream physical outcome if the deviation proceeds unmitigated (e.g., *Reactor overpressurization $\rightarrow$ rupture $\rightarrow$ toxic vapor release*).
- **Safeguard:** An engineered device, control interlock, alarm, or operational procedure designed to prevent the cause or mitigate the consequence (e.g., *TAH High-temperature alarm*, *PSV Pressure Safety Valve*, *Emergency Shutdown System (ESD)*).
- **Recommendation:** Action items proposed when existing safeguards are deemed insufficient for the assessed risk.

### 2.2 The Tabular Representation Paradox
Industrial HAZOP worksheets organize scenarios into flat rows:

| Deviation | Cause | Consequence | Safeguards |
| :--- | :--- | :--- | :--- |
| MORE Temperature | Cooling pump failure | Reactor overpressure | TAH; PSV; ESD |

**The structural flaw:** The flat tabular layout is an artifact of document formatting, not physical reality. In actual chemical plants:
1. **Intermediate physical states are erased:** Cooling pump failure does not instantly produce overpressure. The real physical chain is:
   $$\text{Cooling pump failure} \xrightarrow{} \text{Loss of cooling water} \xrightarrow{} \text{Elevated reactor temp} \xrightarrow{} \text{Thermal runaway} \xrightarrow{} \text{Vapor generation} \xrightarrow{} \text{Overpressurization}$$
2. **Safeguard attachment points are lost:** The table presents safeguards as a comma-separated bag of strings. In reality, safeguards act on specific intermediate points in the chain:
   - *TAH (Temperature Alarm High)* detects and acts on the *temperature rise*.
   - *PSV (Pressure Safety Valve)* acts on the terminal *overpressurization*.
   - *ESD (Emergency Shutdown)* isolates reactant feeds.
3. **Branching and multi-cause convergence are obscured:** A single loss of containment event can branch into direct pool fire, explosive vapor cloud formation, or toxic environmental dispersion. Conversely, multiple independent component failures can converge on identical intermediate deviations.

Flat text mining treats this as a superficial classification problem, missing the multi-hop relational dependencies needed to verify if a consequence or safeguard is chemically and physically valid.

---

## 3. Comprehensive Literature Taxonomy & Prior Art

The project establishes a five-tier taxonomy categorization across literature spanning 1995 to 2026:

```
[A. Classical Models / Digraphs] ──> Digraph traversal on hand-crafted P&ID models
[B. Ontology Reasoners]         ──> Symbolic OWL/DL reasoning over pre-encoded ontologies
[C. Text Mining / NLP]           ──> Classification & NER on report text (no causal graph)
[D. Knowledge Graphs]           ──> KG construction & graph neural completion (HAZOPCT)
[E. Direct LLM Generation]       ──> Multimodal end-to-end prompt generation (low validity)
```

---

### Category A: Classical Model-Based & Expert Systems
- **Seminal Works:** Vaidhyanathan & Venkatasubramanian (1995, 1996), Rahman et al. (*ExpHAZOP+*, 2009), Rodríguez & de la Mata (*D-higraphs*, 2012), Taylor (2017).
- **Mechanism:** Operates on qualitative physics models, signed directed graphs (SDGs), and rule bases constructed by expert systems.
- **Strengths:** Deterministic, physically consistent causal propagation.
- **Critical Limitation:** Extreme manual authoring overhead; cannot parse or learn from historical textual documentation.

---

### Category B: Ontology-Based Symbolic Causal Reasoning (Deep Dive: Single et al., 2020)
- **Primary Source:** Single, J. I., Schmidt, J., & Denecke, J. (2020). *Ontology-based computer aid for the automation of HAZOP studies.* Journal of Loss Prevention in the Process Industries, 68, 104321.
- **System Architecture:** Implemented in OWL 2 DL, executed with the HermiT reasoner via `owlready2` in Python. Evaluated on a **Hexane Storage Tank (T-301)**.
- **Key Findings:**
  - Proves that automated causal multi-hop propagation, branching consequences, and multi-cause convergence are completely feasible via formal ontological reasoning.
  - Generates extended worksheets with 8 explicit causal fields: `Unit`, `Substance`, `Deviation`, `Super Cause`, `Cause`, `Effect`, `Consequence`, `Safeguard`.
  - Generates 40 scenarios across 8 deviations that match human expert worksheets in quality and completeness.
- **The Research Boundary:**
  - Single et al. automate the **reasoning step**, NOT the **knowledge acquisition step**.
  - The process equipment, substance properties (flammability limits, boiling point of hexane), and causal propagation axioms must all be pre-encoded in OWL 2 DL by human experts before HermiT can infer a single link. It does not ingest raw textual HAZOP reports.

---

### Category C: NLP & Statistical Text Mining
- **Primary Sources:** Feng et al. (2021), Wang et al. (2021), Wang & Gu (2022), Ekramipooya et al. (2023, 2024).
- **Mechanism:** Named Entity Recognition (NER), BERT embeddings, and supervised classifiers applied to extract spans or predict categorical deviations from cause strings.
- **Critical Limitation:** Conflates label prediction with causal reasoning. For instance, Ekramipooya et al. (2023) predict deviation categories from cause text across 25 records (42 with SMOTE oversampling). This is correlational pattern matching; it does not build or navigate a causal graph.

---

### Category D: Process Safety Knowledge Graphs (Deep Dive: HAZOPCT & Mao et al.)
- **Primary Sources:** Mao et al. (2020, *Delayed coking KG*), Li & Zhao (2024, 2025, *HAZOPCT*).
- **Mechanism:** Builds heterogeneous Knowledge Graphs from multi-source engineering documents (P&ID, operating manuals, HAZOP records) and applies Graph Neural Networks (Graph Attention Networks / GAT) or link prediction to identify missing safety relationships.
- **Key Findings (HAZOPCT - Li & Zhao, 2025):**
  - Formulates HAZOP completion as link prediction over a predefined ontology evaluated on a sulfur recovery unit.
  - Demonstrates that attention weights can provide interpretability for missing consequence recommendations.
- **The Research Boundary:** Operates primarily on structured multi-source information with fixed ontological nodes, rather than reconstructing open-ended multi-hop causal chains from messy, unstructured historical narrative text.

---

### Category E: Direct LLM Worksheet Generation (Deep Dive: Lee et al., 2026)
- **Primary Source:** Lee, J., Park, S., Oh, S., & Ma, B. (2026). *Can large language models automate the HAZOP process without human intervention?* Safety Science, 194, 107039.
- **Mechanism:** Prompts state-of-the-art multimodal LLMs (GPT-4V, etc.) with P&ID images and standardized instructions to generate full HAZOP tables end-to-end.
- **Crucial Findings:**
  - **High Lexical Similarity:** The generated tables achieved high BLEU, ROUGE, and METEOR scores when compared to human engineering tables.
  - **Low Scenario Validity:** Expert chemical engineering review revealed that only **19% to 37%** of the generated hazard scenarios were physically and semantically valid!
  - **Safeguard Bias:** The LLM exhibited severe hallucinations and systematic bias toward soft procedural recommendations ("train operators", "check manual") rather than engineered hardware safeguards (PSVs, SIS, interlocks).
- **Takeaway for this Project:** Unconstrained LLM text generation is fundamentally unsafe for chemical process safety. LLMs must be bounded by structured causal graph representations, explicit verification, and provenance checks.

---

## 4. The 5-Level Capability Framework & Identified Research Gap

To avoid the ambiguity prevalent in existing literature, this project defines and evaluates systems across five distinct operational capabilities:

1. **Extraction:** Can the system isolate entities, parameters, and tokens from raw textual reports?
2. **Representation:** Does it represent information as an explicit relational graph/ontology, or merely as flat rows?
3. **Causal Reasoning:** Can it navigate intermediate physical states, directional propagation, branching, and multi-cause convergence?
4. **Inference:** Can it propose unstated or missing causes, consequences, or safeguards?
5. **Validation:** Are proposals checked against physical laws, process models, or expert ground truth?

### Literature Capability Matrix

| Literature Category | Primary Input Modality | Extraction | Explicit Repr. | Causal Reasoning | Missing Inference | Formal Validation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A. Classical Digraphs** | Hand-crafted Process Model | $\times$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | Partial |
| **B. Ontology Reasoners** | Hand-crafted OWL Ontology | $\times$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **C. Text Mining / NLP** | HAZOP Textual Reports | $\checkmark$ | $\times$ | $\times$ | $\times$ | Partial |
| **D. Knowledge Graphs** | Structured Multi-Source + KG | Partial | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **E. LLM Generation** | P&ID + Free-form Prompt | N/A | $\times$ | Partial | $\checkmark$ | $\checkmark$ (Failed: 19–37%) |
| **Target of this Project** | **Unstructured Historical Reports** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | **$\checkmark$ (Constrained)** |

### The Exact Research Gap
No prior work reconstructs typed, directional causal graphs directly from **unstructured historical HAZOP documentation** and uses that extracted graph to **rigorously constrain multi-hop inference** while enforcing explicit provenance and abstention mechanisms.

---

## 5. The Working Research Hypothesis & Proposed Pipeline

The project investigates an end-to-end framework that takes messy historical reports and produces verifiable, multi-hop causal reasoning:

```
┌─────────────────────────────────┐
│ Historical HAZOP Text / CSVs    │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ LLM / NLP Causal Extraction     │
│ (Entities & Typed Propositions) │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Typed Causal Graph  G = (V, E)  │
│ (Nodes, Intermediate States)    │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Bounded Multi-Hop Reasoner      │
│ (Direct Edges & Path Tracing)   │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Process Safety Validation & KG  │
│ Consistency (Abstain if Ambigu.)│
└─────────────────────────────────┘
```

### 5.1 Formal Graph Representation Schema
The safety knowledge is modeled as a directed graph $G = (V, E)$:
- **Entity Types ($V$):**
  - `Process Node`: Physical boundary (e.g., *Hexane Storage Unit*).
  - `Equipment`: Specific hardware (e.g., *Pump P-101*, *Tank T-301*, *Valve V-12*).
  - `Parameter`: Process variable (e.g., *Pressure*, *Temperature*, *Level*).
  - `Deviation`: Standard deviation pair (e.g., *High Level*, *More Temperature*).
  - `Cause`: Initiating malfunction or operational error.
  - `Effect`: Intermediate physical state or phenomenon (e.g., *Excess Inflow*, *Overfilling*).
  - `Consequence`: Terminal hazardous outcome (e.g., *Loss of Primary Containment*, *Fire*, *Vapor Cloud Explosion*).
  - `Safeguard`: Active/passive mitigation layer (e.g., *PSV*, *High Level Switch*, *Bund Wall*).
  - `Recommendation`: Proposed action item.

### 5.2 Causal Proposition Formulation
Relationships in $E$ are represented as directed triples $(h, r, t)$ accompanied by epistemic metadata:
- `causes_deviation`: $\text{Cause} \xrightarrow{} \text{Deviation}$
- `leads_to`: $\text{Deviation} \xrightarrow{} \text{Intermediate Effect}$
- `leads_to_consequence`: $\text{Intermediate Effect} \xrightarrow{} \text{Consequence}$
- `mitigated_by`: $\text{Consequence / State} \xrightarrow{} \text{Safeguard}$
- `has_parameter`, `has_deviation`: Structural topology links.

---

## 6. Software Architecture & Technical Implementation (`server/`)

The software implementation in [`server/app`](file:///mnt/volume/Projects/BTech-Project/NLP-HAZOP/server/app) follows **Clean Architecture** and **Domain-Driven Design (DDD)** principles to guarantee complete decoupling between chemical engineering domain rules, reasoning algorithms, and infrastructure frameworks.

```
server/app/
├── domain/                  # Enterprise business logic & pure models
│   ├── entities/            # hazop_record.py (CausalRelation, RecordType, Origin)
│   ├── reasoning/           # models.py (ReasoningResult, AbstentionReason)
│   └── repositories/        # graph_repository.py (Abstract interfaces)
├── application/             # Application orchestration & use cases
│   ├── strategies/          # Reasoners: Graph, TF-IDF, Prompt-Only, Hybrid
│   │   └── retrieval/       # Sparse vector & neighborhood extractors
│   ├── use_cases/           # recover_missing_link.py, build_evidence_context.py
│   └── services/            # hazop_service.py
├── infrastructure/          # External frameworks, parsers, DB adapters
│   ├── datasets/            # hazop_csv_loader.py (Provenance-enforcing parser)
│   └── graph/               # in_memory_graph_repository.py
├── api/                     # Presentation layer
│   └── routes/              # imports.py (FastAPI endpoints)
└── main.py                  # ASGI FastAPI application factory
```

### 6.1 Clean Architecture / Domain-Driven Design (DDD)
1. **Domain Layer:** Contains zero third-party dependencies (no FastAPI, no Pandas, no PyTorch, no Groq). It defines pure dataclasses: [`CausalRelation`](file:///mnt/volume/Projects/BTech-Project/NLP-HAZOP/server/app/domain/entities/hazop_record.py#L76), [`HAZOPRecord`](file:///mnt/volume/Projects/BTech-Project/NLP-HAZOP/server/app/domain/entities/hazop_record.py#L112), and protocols.
2. **Application Layer:** Contains algorithmic reasoning strategies implementing a common protocol.
3. **Infrastructure Layer:** Implements concrete repositories (in-memory graph stores, CSV parsers).
4. **API Layer:** Exposes RESTful endpoints via FastAPI.

### 6.2 Domain Models & Strict Provenance Tracking
Every record and relation in the system carries an explicit epistemic origin:

```python
class RelationshipOrigin(StrEnum):
    SOURCE = "source"                      # Verbatim source row
    RECONSTRUCTED = "reconstructed"        # Reconstructed from literature case
    SYNTHETIC = "synthetic"                # Simulation or generator derived
    EXPERT = "expert"                      # Human expert adjudicated
    MODEL_INFERRED = "model_inferred"      # LLM generated proposal
    GRAPH_INFERRED = "graph_inferred"      # Bounded multi-hop traversal hypothesis
    RETRIEVAL_INFERRED = "retrieval_inferred" # Lexical row co-occurrence candidate
```

Crucially, **the loader never fabricates high confidence or promotes unreviewed records**. If an imported row says "Review required", its validation status remains `UNREVIEWED`.

### 6.3 Reasoning & Retrieval Strategies
The codebase implements four distinct missing-endpoint reasoning engines:

1. **[`PromptOnlyReasoner`](file:///mnt/volume/Projects/BTech-Project/NLP-HAZOP/server/app/application/strategies/prompt_only_reasoner.py):**
   - Given only the held-out scenario context (Node, Deviation, Cause), prompts an LLM via Groq API to propose missing consequences or safeguards.
   - Evaluates whether raw parametric LLMs can predict outcomes without domain graph retrieval.
2. **[`TfidfRetriever`](file:///mnt/volume/Projects/BTech-Project/NLP-HAZOP/server/app/application/strategies/retrieval/tfidf.py):**
   - Transparent, sparse lexical baseline.
   - Builds TF-IDF indices across training scenarios and ranks candidate endpoints based on lexical overlap.
3. **[`GraphReasoner`](file:///mnt/volume/Projects/BTech-Project/NLP-HAZOP/server/app/application/strategies/graph_reasoner.py):**
   - Implements bounded Breadth-First Search (BFS) path search up to $k$ hops (default: 4).
   - Prefers direct evidence matching the predicate.
   - If direct evidence is missing, searches for typed causal paths:
     $$\text{Cause} \xrightarrow{\text{causes\_deviation}} \text{Deviation} \xrightarrow{\text{leads\_to}} \text{Effect} \xrightarrow{\text{leads\_to\_consequence}} \text{Consequence}$$
   - **Safety Abstention Logic:** If multiple competing terminal endpoints are found with conflicting evidence, or if support is insufficient, the reasoner actively **abstains** with `AbstentionReason.AMBIGUOUS_EVIDENCE` or `INSUFFICIENT_SUPPORT` rather than guessing.
4. **[`HybridReasoner`](file:///mnt/volume/Projects/BTech-Project/NLP-HAZOP/server/app/application/strategies/hybrid_reasoner.py):**
   - Assembles candidates from both TF-IDF retrieval and graph traversal.
   - Presents the structured provenance and path evidence to an LLM reasoner for adjudication.

### 6.4 FastAPI Service & CSV Preview Pipeline
The API layer in [`server/app/api/routes/imports.py`](file:///mnt/volume/Projects/BTech-Project/NLP-HAZOP/server/app/api/routes/imports.py) provides a high-throughput endpoint:
- `POST /api/v1/datasets/preview`: Accepts UTF-8 HAZOP CSV files (up to 10 MB), validates standard HAZOP headers, normalizes columns (`Scenario_ID`, `Deviation`, `Possible_Causes`, `Possible_Consequences`), and returns normalized records with explicit provenance warnings indicating that row-level associations are unverified.

---

## 7. Datasets, Provenance Rules & Epistemic Hygiene

The project maintains two distinct datasets partitioned into three isolated evaluation tracks. Pooling them is strictly prohibited.

```
data/
└── raw/
    ├── t301/
    │   └── T301_HAZOP_CCPS2001_style.csv     # 40 Reconstructed + 54 Synthetic
    └── dwsim/
        ├── synthetic_hazop_D001_D030.csv     # 30 Synthetic simulation records
        └── HAZOP_DWSIM30_worksheet_format.csv # Alternate format of D001-D030
```

### 7.1 T-301 Hexane Storage Tank Track
- **Origin:** Reconstructed from the hexane storage case reported in Single et al. (2020).
- **Track 1 (`t301-reconstructed`):** 40 records representing structured scenario chains derived from the paper's reported case.
- **Track 2 (`t301-synthetic`):** 54 synthetic continuation records generated to test scale.

### 7.2 DWSIM Ethanol / Fermentation Plant Track
- **Origin:** Rigorous process simulations conducted in DWSIM for an industrial ethanol fermentation plant.
- **Track 3 (`dwsim-synthetic`):** 30 scenarios (D001–D030) capturing physical perturbations (e.g., cooling jacket failure, valve sticking, temperature deviations).
- **Caveats:** Retains raw simulation notes: 16 *"Review required"*, 6 *"Scope caveat"*, 6 *"NEEDS_MODEL_REVISION"*, 1 *"Review reaction model"*, 1 *"NEEDS_REVIEW"*.

### 7.3 The Silver Label Caveat
The dataset labels are **silver row associations**:
- If a row lists `Cause: Pump trip` and `Consequence: Overfill; Fire; Explosion`, the evaluation treats those consequences as a silver target set for that row.
- **Epistemic Constraint:** This row co-occurrence does not prove that the specific cause strictly produces every listed consequence in isolation. True validation requires formal chemical engineer adjudication.

---

## 8. Leakage-Aware Benchmark & Baseline Findings

### 8.1 Benchmark Design
Implemented in [`server/scripts/run_benchmark.py`](file:///mnt/volume/Projects/BTech-Project/NLP-HAZOP/server/scripts/run_benchmark.py):
- **Scenario-Level Holdout:** Uses a strict 80/20 train/test split (seed 42). Entire scenarios are withheld to ensure no scenario-level data leaks into the training graph or TF-IDF index.
- **Query Generation:** For each held-out scenario, all consequence/safeguard fields are stripped. One test query is issued per cause.
- **Silver Target:** The consequences co-listed in that held-out row.

### 8.2 Baseline Run Results (30 October 2026)
*(From GitHub Actions Run 41048488263; 122 unit tests passing)*

#### Sparse TF–IDF Retrieval Performance
| Metric | T-301 Reconstructed | T-301 Synthetic | DWSIM Synthetic |
| :--- | :---: | :---: | :---: |
| **Micro Precision** | 26.80% | 0.00% | 9.00% |
| **Micro Recall** | 47.27% | 0.00% | 34.62% |
| **Micro F1** | **34.21%** | 0.00% | **14.29%** |
| **Hit@1** | 15.79% | 0.00% | 26.67% |
| **MRR** | 0.3456 | 0.0000 | 0.3278 |
| **Coverage** | 89.47% | 100.00% | 93.33% |
| **Abstention Rate** | 10.53% | 0.00% | 6.67% |
| **Mean Query Latency** | 0.151 ms | 0.371 ms | 0.131 ms |

#### Graph Reasoning Performance
| Metric | T-301 Reconstructed | T-301 Synthetic | DWSIM Synthetic |
| :--- | :---: | :---: | :---: |
| **Micro F1** | 0.00% | 0.00% | 0.00% |
| **Coverage** | 0.00% | 0.00% | 0.00% |
| **Abstention Rate** | **100.00%** | **100.00%** | **100.00%** |
| *Ambiguous-evidence abstentions* | 17 | 0 | 0 |
| *Insufficient-support abstentions* | 2 | 11 | 15 |

### 8.3 Analysis of Results & Abstention Dynamics
1. **The Lexical Baseline Paradox:** TF-IDF achieved 89–100% coverage because lexical matching will almost always retrieve *something* from the training set. However, on T-301 synthetic, its precision was 0.00% despite 100% coverage, demonstrating that high coverage in safety systems without semantic grounding is meaningless and hazardous.
2. **Graph Abstention as a Safety Feature:** The Graph Reasoner abstained 100% of the time. In the T-301 reconstructed track, **17 out of 19 queries abstained due to ambiguous evidence**. Because multiple historical rows shared identical cause nodes leading to conflicting consequence branches, the reasoner refused to flip an arbitrary coin.
3. **Graph Coverage Gap:** In synthetic tracks, the held-out causes lacked exact lexical node matches in the training subgraph. This establishes the immediate research requirement: implementing **ontological entity normalization** and **dense semantic linking** so that the graph reasoner can recognize synonymous process concepts while preserving its strict ambiguity-abstention safety behavior.

---

## 9. Open Research Questions & Experimental Roadmap

### 9.1 The Primary Research Questions
1. **RQ1 (Causal Extraction Fidelity):** Can modern LLMs extract typed causal triples $(h, r, t)$ from historical narrative worksheets with precision and recall exceeding classical NLP pipelines?
2. **RQ2 (Multi-Hop Superiority):** Does an explicit graph representation improve multi-hop hazard recovery accuracy and reduce hallucination compared to direct prompt-based LLM generation?
3. **RQ3 (Safety-Critical Abstention):** Can a hybrid reasoner leverage graph-provenance constraints to maintain high accuracy when confident while maintaining a defensible abstention rate under ambiguous evidence?

### 9.2 Immediate Experimental Roadmap
- [ ] **Run Prompt-Only & Hybrid LLM Baselines:** Execute benchmarks across identical held-out splits with Groq API integration (Llama 3 / Mixtral), logging token consumption, costs, and latency.
- [ ] **Ontological Concept Normalization:** Implement entity canonicalization (e.g., mapping `"Tank overfilling"` and `"Liquid overflow"` to an identical ontology node) to resolve graph sparsity without relaxing causal path constraints.
- [ ] **Expert Adjudication of Silver Labels:** Partner with chemical engineering safety specialists to adjudicate a gold evaluation benchmark, establishing true causal ground truth.
- [ ] **Dense Embedding Integration:** Benchmark dense vector retrievers against the current sparse TF-IDF baseline.
- [ ] **Cross-Plant Generalization:** Evaluate whether causal rules extracted from storage units (T-301) can transfer or generalize to reaction and separation units (DWSIM).

---

*This document serves as the authoritative technical and theoretical research reference for the `NLP-HAZOP` project.*
