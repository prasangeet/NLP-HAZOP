const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export interface Dataset {
  dataset_id: string;
  name: string;
  filename: string;
  record_count: number;
  relationship_count: number;
  created_at: string;
}

export interface Scenario {
  scenario_id: string;
  equipment?: string;
  deviation?: string;
  causes?: string[];
  effects?: string[];
  consequences?: string[];
  safeguards?: string[];
  recommendations?: string[];
  [key: string]: unknown;
}

export interface Relationship {
  subject: string;
  predicate: string;
  object: string;
  origin: string;
  evidence?: string | null;
  confidence?: number | null;
  source_id?: string | null;
  is_expert_verified: boolean;
  scenario_id?: string;
}

export interface GraphNode {
  id: string;
  label: string;
  entity_type: string;
  dataset_id: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  predicate: string;
  scenario_id: string;
  origin: string;
  evidence?: string | null;
  is_expert_verified: boolean;
}

export interface GraphResponse {
  dataset_id: string;
  node_count: number;
  edge_count: number;
  truncated: boolean;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface CsvPreview {
  status: string;
  filename: string;
  dataset_id: string;
  storage: string;
  persisted: boolean;
  recognized_columns: Record<string, string>;
  unrecognized_columns: string[];
  missing_expected_columns: string[];
  summary: {
    rows_read: number;
    records_loaded: number;
    rows_skipped: number;
    malformed_rows: number;
    relationships_generated: number;
  };
  warnings: string[];
  preview_limit: number;
  preview_truncated: boolean;
  records: Scenario[];
}

export interface RecoveryResponse {
  dataset_id: string;
  scenario_id: string;
  strategy: string;
  recovery: {
    status: string;
    candidates: Array<Record<string, unknown>>;
    [key: string]: unknown;
  };
  reasoning: {
    proposed_relation?: Record<string, unknown> | null;
    [key: string]: unknown;
  };
  validation: {
    valid: boolean;
    errors: string[];
    warnings: string[];
  };
  notice: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new Error(
      `Cannot reach the NLP-HAZOP API at ${API_BASE_URL}. Start FastAPI and check VITE_API_BASE_URL.`,
    );
  }

  if (!response.ok) {
    let message = `API request failed (${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown; message?: string };
      if (typeof body.detail === "string") message = body.detail;
      else if (body.detail && typeof body.detail === "object") message = JSON.stringify(body.detail);
      else if (body.message) message = body.message;
    } catch {
      // Preserve the HTTP status message for non-JSON responses.
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

function formData(file: File, fields: Record<string, string> = {}): FormData {
  const form = new FormData();
  form.append("file", file);
  for (const [key, value] of Object.entries(fields)) form.append(key, value);
  return form;
}

export const api = {
  baseUrl: API_BASE_URL,
  health: () => request<{ status: string; service: string }>("/health"),
  listDatasets: () => request<{ storage: string; count: number; datasets: Dataset[] }>("/api/v1/datasets"),
  previewDataset: (file: File) => request<CsvPreview>("/api/v1/datasets/preview", { method: "POST", body: formData(file) }),
  importDataset: (file: File, datasetId?: string, name?: string) =>
    request<{ status: string; persisted: boolean; dataset: Dataset }>("/api/v1/datasets/import", {
      method: "POST",
      body: formData(file, { ...(datasetId ? { dataset_id: datasetId } : {}), ...(name ? { name } : {}) }),
    }),
  listScenarios: (datasetId: string) =>
    request<{ dataset_id: string; count: number; scenarios: Scenario[] }>(`/api/v1/datasets/${encodeURIComponent(datasetId)}/scenarios?limit=500`),
  listRelationships: (datasetId: string) =>
    request<{ dataset_id: string; count: number; relationships: Relationship[] }>(`/api/v1/datasets/${encodeURIComponent(datasetId)}/relationships?limit=500`),
  graph: (datasetId: string) =>
    request<GraphResponse>(`/api/v1/datasets/${encodeURIComponent(datasetId)}/graph?limit=1000`),
  extractReport: (file: File, datasetId: string, name?: string) =>
    request<Record<string, unknown>>("/api/v1/reports/extract", { method: "POST", body: formData(file, { dataset_id: datasetId, ...(name ? { name } : {}) }) }),
  runBenchmarkSuite: (payload: {
    datasets: Array<"t301-reconstructed" | "t301-synthetic" | "dwsim-synthetic">;
    seeds: number[];
    test_fraction: number;
    with_llm: boolean;
    max_llm_cases: number;
  }) =>
    request<BenchmarkSuiteRun>("/api/v1/experiments/benchmark-suite", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  listBenchmarkRuns: () =>
    request<{ count: number; runs: BenchmarkSuiteRun[] }>("/api/v1/experiments/benchmark-runs"),
  runBenchmark: (payload: {
    dataset: "t301-reconstructed" | "t301-synthetic" | "dwsim-synthetic";
    seed: number;
    test_fraction: number;
    with_llm: boolean;
    max_llm_cases: number;
  }) =>
    request<BenchmarkRun>("/api/v1/experiments/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  recover: (datasetId: string, payload: {
    scenario_id: string;
    predicate: string;
    known_endpoint: string;
    missing_endpoint: "subject" | "object";
    limit: number;
    strategy: "graph" | "llm" | "hybrid";
  }) =>
    request<RecoveryResponse>(`/api/v1/datasets/${encodeURIComponent(datasetId)}/recovery`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};

export interface BenchmarkMethodMetrics {
  status?: string;
  n_evaluated?: number;
  n_total_test_cases?: number;
  micro_precision?: number | null;
  micro_recall?: number | null;
  micro_f1?: number | null;
  macro_f1?: number | null;
  hit_at_1?: number | null;
  mrr?: number | null;
  coverage?: number | null;
  abstention_rate?: number | null;
  mean_latency_ms?: number | null;
  cost_usd?: number | null;
  [key: string]: unknown;
}

export interface BenchmarkCaseResult {
  case_id: string;
  scenario_id: string;
  record_type: string;
  known_endpoint: string;
  gold_values: string[];
  methods: Record<string, {
    status?: string;
    predictions?: Array<{ value: string; origin?: string; evidence?: string | null }>;
    latency_ms?: number;
    [key: string]: unknown;
  }>;
}

export interface BenchmarkRun {
  status: string;
  dataset: string;
  seed: number;
  test_fraction: number;
  metadata: Record<string, unknown>;
  method_summary: Record<string, BenchmarkMethodMetrics>;
  cases: BenchmarkCaseResult[];
  artifacts: {
    summary_csv: string;
    predictions_csv: string;
    report_md: string;
  };
}


export interface BenchmarkSuiteRun {
  run_id: string;
  created_at: string;
  protocol_version: string;
  label_status: string;
  label_warning: string;
  datasets: string[];
  seeds: number[];
  test_fraction: number;
  with_llm: boolean;
  runs: Array<{
    dataset: string;
    seed: number;
    status: string;
    metadata: Record<string, unknown>;
    method_summary: Record<string, BenchmarkMethodMetrics>;
    case_count: number;
    artifacts: Record<string, string>;
  }>;
  failures: Array<{ dataset: string; seed: number; error: string }>;
}
