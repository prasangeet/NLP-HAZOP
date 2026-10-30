# Repository architecture

This project is a research prototype for extracting typed HAZOP knowledge from reports, storing it with provenance, and evaluating evidence-bounded missing-link recovery.

## Folder map

```text
server/
├── app/
│   ├── api/
│   │   ├── dependencies.py        # dependency injection for repositories and AI providers
│   │   └── routes/                # HTTP contracts: datasets, imports, recovery, extraction
│   ├── application/
│   │   ├── factories/             # composition of retrieval and reasoning strategies
│   │   ├── ports/                 # interfaces required by application use cases
│   │   ├── strategies/
│   │   │   ├── retrieval/          # scenario-local, TF-IDF, hybrid, graph neighborhood
│   │   │   ├── graph_reasoner.py
│   │   │   ├── llm_reasoner.py
│   │   │   ├── hybrid_reasoner.py
│   │   │   └── prompt_only_reasoner.py  # independent benchmark baseline
│   │   └── use_cases/              # extract/store, retrieve, build evidence context
│   ├── domain/
│   │   ├── entities/               # HAZOP records, datasets, extracted report models
│   │   ├── ontology/               # entity types and allowed predicate signatures
│   │   ├── repositories/           # repository interfaces
│   │   └── services/               # deterministic validation
│   ├── infrastructure/
│   │   ├── datasets/               # CSV import adapter
│   │   ├── extraction/             # Groq/Llama report extraction adapter
│   │   ├── factories/              # concrete AI provider factory
│   │   └── graph/                  # in-memory and Neo4j repository adapters
│   └── main.py                     # FastAPI composition root
└── tests/
    ├── unit/
    └── integration/
```

## Design patterns used

| Pattern | Where it is used | Responsibility |
|---|---|---|
| Repository | `domain/repositories/graph_repository.py` | Defines storage operations without tying the domain to Neo4j |
| Repository adapters | `infrastructure/graph/` | Implements the same contract with in-memory and Neo4j storage |
| Strategy | `application/strategies/` | Makes graph, evidence-based LLM, hybrid and retrieval methods interchangeable |
| Abstract Factory | `AIProviderFactory`, `GroqAIProviderFactory` | Creates a compatible set of AI extraction and reasoning adapters |
| Factory Method / composition factory | `ApplicationFactory` | Selects and wires retrieval and reasoning strategies for an API request |
| Ports and Adapters | `ReportExtractor` port + `GroqReportExtractor` adapter | Keeps the extraction use case independent from the Groq SDK |
| Dependency Injection | `api/dependencies.py` and route parameters | Supplies repository and AI-provider dependencies at the HTTP boundary |

The application layer depends on interfaces and domain objects. Neo4j, Groq, FastAPI and file upload handling stay in infrastructure/API layers.

## Data flow

### Report extraction and persistence

```text
TXT / MD / LOG upload
        |
        v
GroqReportExtractor (structured extraction)
        |
        v
ExtractAndStoreReport
        |
        +----> HazopSafetyValidator
        |        - canonical predicate/type signatures
        |        - source quote must occur verbatim in input
        |        - scenario fields must agree with extracted edges
        |
        v
DatasetRepository
        |
        v
Neo4j (typed nodes, typed edges, source quote, origin, source ID)
```

Extracted edges are stored with origin `llm_extracted`, validation status `unreviewed`, and the exact source quote. The validator checks structure and quote presence; it does not establish physical causality or provide expert approval.

### Missing-link recovery

```text
Recovery request
      |
      v
ScenarioLocalRetriever ---- TF-IDF Retriever
      |                         |
      +------ HybridRetriever --+
                    |
             GraphNeighborhoodRetriever
                    |
          +---------+--------------------+
          |                              |
     evidence_items                 graph_context
  (direct/TF-IDF candidates)     (actual graph edges only)
          |                              |
          |                       GraphReasoner
          |                              |
          |                       typed path proposal
          +---------------+--------------+
                          |
                 selected strategy
             graph / LLM / hybrid
                          |
                 HazopSafetyValidator
                          |
                proposal or abstention
```

The two context collections are intentionally separate. TF-IDF row matches are candidate evidence, not graph edges, and must never be passed to graph traversal as if they were explicit relationships. Hybrid reasoning runs graph traversal only over `graph_context`, then gives the LLM the candidates and any graph-path hypothesis for adjudication.

## Canonical ontology

Supported entity types: `ProcessUnit`, `Equipment`, `Parameter`, `Cause`, `Deviation`, `Effect`, `Consequence`, `Safeguard`, `Recommendation`, and fallback `Concept`.

| Predicate | Subject | Object | Meaning |
|---|---|---|---|
| `causes_deviation` | Cause | Deviation | Cause associated with a deviation |
| `leads_to` | Deviation | Effect | Deviation propagates to an intermediate effect |
| `leads_to_consequence` | Effect or Deviation | Consequence | An effect/deviation is associated with a consequence |
| `co_occurs_with_consequence` | Deviation | Consequence | Non-causal row association when an intermediate-effect pairing is ambiguous |
| `mitigated_by` | Deviation, Effect or Consequence | Safeguard | Mitigation association |
| `recommends_action` | Deviation or Consequence | Recommendation | Recommended action association |
| `has_parameter` | Equipment | Parameter | Equipment parameter |
| `located_in` | Equipment | ProcessUnit | Equipment location |

For a row containing multiple intermediate effects, the importer does not pair effects to consequences by position or Cartesian product. It records deviation/consequence co-occurrence instead. The graph reasoner composes only the typed propagation chain `leads_to` → `leads_to_consequence`; it does not invent a direct cause-to-consequence edge.

## Running

From the repository root, start Neo4j:

```bash
cp .env.example .env
# Set a local password and GROQ_API_KEY if LLM functionality is required.
docker compose up -d
```

Then start the API:

```bash
cd server
uv sync
uv run uvicorn app.main:app --reload --env-file ../.env --host 127.0.0.1 --port 8000
```

The extraction endpoint accepts UTF-8 `.txt`, `.md`, or `.log` files up to 2 MiB at `POST /api/v1/reports/extract`. PDF ingestion is not implemented in this version.

## Validation

```bash
cd server
uv run ruff check app tests scripts
uv run pytest -v
```

The CI workflow also runs the dataset benchmark suite and a real Neo4j integration test using a disposable Docker database.

## Research and safety boundaries

- The available CSV benchmarks are reconstructed or synthetic and use row-derived silver labels. They are not expert-validated causal ground truth.
- TF-IDF is lexical sparse retrieval, not a dense embedding model.
- Extracted, retrieved, reconstructed, synthetic and graph-inferred propositions retain their provenance and are not promoted to expert-validated facts.
- Graph paths express typed reachability hypotheses; they do not establish that the two endpoints form an immediate causal edge.
- Model confidence is not a calibrated probability.
- This software is a research aid, not an operational HAZOP system, engineering design validator, or safety approval tool.
