import { useState } from "react";
import {
  Activity,
  CheckCircle2,
  CircleAlert,
  Clock3,
  FlaskConical,
  GitBranch,
  LoaderCircle,
  Play,
  RotateCcw,
  ShieldCheck,
  Workflow,
  XCircle,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import PageHeader from "@/components/layout/PageHeader";
import { api, type BenchmarkRun, type BenchmarkSuiteRun, type RecoveryResponse } from "@/lib/api";
import { useWorkspaceStore } from "@/lib/workspace-store";

type CheckStatus = "idle" | "running" | "passed" | "warning" | "failed";

type CheckResult = {
  id: string;
  title: string;
  status: CheckStatus;
  summary: string;
  details: string[];
  durationMs?: number;
};

type StrategyResult = {
  strategy: "graph" | "llm" | "hybrid";
  status: "passed" | "failed";
  durationMs: number;
  candidateCount: number;
  response: RecoveryResponse | null;
  error?: string;
};

const strategyLabels = {
  graph: "Graph-based",
  llm: "LLM-based",
  hybrid: "Hybrid",
} as const;

function formatBenchmarkMetric(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(3);
}

function StatusBadge({ status }: { status: CheckStatus }) {
  if (status === "passed") return <Badge variant="secondary">Passed</Badge>;
  if (status === "failed") return <Badge variant="destructive">Failed</Badge>;
  if (status === "warning") return <Badge variant="outline">Warning</Badge>;
  if (status === "running") return <Badge variant="outline">Running</Badge>;
  return <Badge variant="outline">Not run</Badge>;
}

function ResultIcon({ status }: { status: CheckStatus }) {
  if (status === "passed") return <CheckCircle2 className="size-4 text-primary" />;
  if (status === "failed") return <XCircle className="size-4 text-destructive" />;
  if (status === "warning") return <CircleAlert className="size-4 text-muted-foreground" />;
  if (status === "running") return <LoaderCircle className="size-4 animate-spin text-muted-foreground" />;
  return <Activity className="size-4 text-muted-foreground" />;
}

export default function Experiments() {
  const activeDatasetId = useWorkspaceStore((state) => state.activeDatasetId);
  const [checks, setChecks] = useState<CheckResult[]>([
    {
      id: "api",
      title: "API connectivity",
      status: "idle",
      summary: "Check whether the FastAPI service responds.",
      details: [],
    },
    {
      id: "dataset",
      title: "Dataset consistency",
      status: "idle",
      summary: "Compare dataset, scenario, and relationship data returned by the API.",
      details: [],
    },
    {
      id: "graph",
      title: "Knowledge graph integrity",
      status: "idle",
      summary: "Check graph nodes, edges, and relationship endpoints.",
      details: [],
    },
  ]);
  const [strategyResults, setStrategyResults] = useState<StrategyResult[]>([]);
  const [running, setRunning] = useState(false);
  const [lastRun, setLastRun] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [benchmarkDataset, setBenchmarkDataset] = useState<BenchmarkRun["dataset"]>("t301-reconstructed");
  const [benchmarkSeed, setBenchmarkSeed] = useState(42);
  const [benchmarkTestFraction, setBenchmarkTestFraction] = useState(0.2);
  const [benchmarkWithLlm, setBenchmarkWithLlm] = useState(false);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkRun | null>(null);
  const [suiteDatasets, setSuiteDatasets] = useState<Array<"t301-reconstructed" | "t301-synthetic" | "dwsim-synthetic">>(["t301-reconstructed", "t301-synthetic", "dwsim-synthetic"]);
  const [suiteSeeds, setSuiteSeeds] = useState("42, 7, 21");
  const [suiteRunning, setSuiteRunning] = useState(false);
  const [suiteError, setSuiteError] = useState<string | null>(null);
  const [suiteResult, setSuiteResult] = useState<BenchmarkSuiteRun | null>(null);
  const [savedRuns, setSavedRuns] = useState<BenchmarkSuiteRun[]>([]);


  const runResearchBenchmark = async () => {
    if (benchmarkRunning) return;
    setBenchmarkRunning(true);
    setBenchmarkError(null);
    setBenchmarkResult(null);
    try {
      const result = await api.runBenchmark({
        dataset: benchmarkDataset,
        seed: benchmarkSeed,
        test_fraction: benchmarkTestFraction,
        with_llm: benchmarkWithLlm,
        max_llm_cases: 15,
      });
      setBenchmarkResult(result);
    } catch (error) {
      setBenchmarkError(error instanceof Error ? error.message : "Benchmark execution failed.");
    } finally {
      setBenchmarkRunning(false);
    }
  };

  const runBenchmarkSuite = async () => {
    if (suiteRunning) return;
    const seeds = suiteSeeds.split(",").map((value) => Number(value.trim()));
    if (!suiteDatasets.length || !seeds.length || seeds.some((seed) => !Number.isInteger(seed) || seed < 0 || seed > 1000000) || new Set(seeds).size !== seeds.length) {
      setSuiteError("Select at least one dataset and enter unique integer seeds between 0 and 1000000.");
      return;
    }
    setSuiteRunning(true); setSuiteError(null); setSuiteResult(null);
    try {
      const result = await api.runBenchmarkSuite({ datasets: suiteDatasets, seeds, test_fraction: benchmarkTestFraction, with_llm: benchmarkWithLlm, max_llm_cases: 15 });
      setSuiteResult(result);
      const history = await api.listBenchmarkRuns();
      setSavedRuns(history.runs);
    } catch (error) {
      setSuiteError(error instanceof Error ? error.message : "Benchmark suite failed.");
    } finally { setSuiteRunning(false); }
  };

  const downloadBenchmarkArtifact = (filename: string, content: string, mimeType: string) => {
    const url = URL.createObjectURL(new Blob([content], { type: mimeType }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const setCheck = (result: CheckResult) => {
    setChecks((current) => {
      const found = current.some((item) => item.id === result.id);
      return found
        ? current.map((item) => (item.id === result.id ? result : item))
        : [...current, result];
    });
  };

  const runConnectivity = async () => {
    const started = performance.now();
    setCheck({
      id: "api",
      title: "API connectivity",
      status: "running",
      summary: "Checking FastAPI health endpoint…",
      details: [],
    });
    try {
      const health = await api.health();
      setCheck({
        id: "api",
        title: "API connectivity",
        status: health.status?.toLowerCase() === "ok" || Boolean(health.status)
          ? "passed"
          : "warning",
        summary: `Health endpoint responded: ${health.status || "status not provided"}.`,
        details: [`Service: ${health.service || "not specified"}`, `Endpoint: ${api.baseUrl}/health`],
        durationMs: Math.round(performance.now() - started),
      });
      return true;
    } catch (error) {
      setCheck({
        id: "api",
        title: "API connectivity",
        status: "failed",
        summary: error instanceof Error ? error.message : "Health check failed.",
        details: [`Endpoint: ${api.baseUrl}/health`],
        durationMs: Math.round(performance.now() - started),
      });
      return false;
    }
  };

  const runDatasetChecks = async () => {
    const started = performance.now();
    setCheck({
      id: "dataset",
      title: "Dataset consistency",
      status: "running",
      summary: "Loading dataset records, scenarios, and relationships…",
      details: [],
    });
    setCheck({
      id: "graph",
      title: "Knowledge graph integrity",
      status: "running",
      summary: "Checking graph node and edge references…",
      details: [],
    });

    if (!activeDatasetId) {
      const message = "Select an imported dataset on the Datasets page first.";
      setCheck({
        id: "dataset",
        title: "Dataset consistency",
        status: "warning",
        summary: message,
        details: [],
      });
      setCheck({
        id: "graph",
        title: "Knowledge graph integrity",
        status: "warning",
        summary: message,
        details: [],
      });
      return null;
    }

    try {
      const [datasetResponse, scenariosResponse, relationshipsResponse, graph] =
        await Promise.all([
          api.listDatasets(),
          api.listScenarios(activeDatasetId),
          api.listRelationships(activeDatasetId),
          api.graph(activeDatasetId),
        ]);

      const dataset = datasetResponse.datasets.find(
        (item) => item.dataset_id === activeDatasetId,
      );
      const scenarios = scenariosResponse.scenarios ?? [];
      const relationships = relationshipsResponse.relationships ?? [];
      const nodeIds = new Set(graph.nodes.map((node) => node.id));
      const danglingEdges = graph.edges.filter(
        (edge) => !nodeIds.has(edge.source) || !nodeIds.has(edge.target),
      );
      const relationshipCountMismatch =
        typeof dataset?.relationship_count === "number" &&
        dataset.relationship_count !== relationships.length;
      const datasetDetails = [
        `Dataset: ${dataset?.name ?? activeDatasetId}`,
        `Scenarios returned: ${scenarios.length}`,
        `Relationships returned: ${relationships.length}`,
        dataset
          ? `Dataset summary count: ${dataset.relationship_count}`
          : "Dataset ID was not present in the dataset listing.",
      ];
      const datasetPassed = Boolean(dataset) && scenarios.length > 0;
      setCheck({
        id: "dataset",
        title: "Dataset consistency",
        status: !dataset ? "failed" : relationshipCountMismatch ? "warning" : datasetPassed ? "passed" : "warning",
        summary: !dataset
          ? "The selected dataset was not found in the dataset listing."
          : relationshipCountMismatch
            ? "The dataset summary count differs from the relationship endpoint count; check endpoint semantics or pagination."
            : datasetPassed
              ? "Dataset and scenario records are accessible."
              : "Dataset is accessible, but no scenarios were returned.",
        details: datasetDetails,
        durationMs: Math.round(performance.now() - started),
      });

      const graphPassed = graph.nodes.length > 0 && danglingEdges.length === 0;
      setCheck({
        id: "graph",
        title: "Knowledge graph integrity",
        status: danglingEdges.length > 0 ? "failed" : graphPassed ? "passed" : "warning",
        summary: danglingEdges.length > 0
          ? `${danglingEdges.length} edge(s) reference missing nodes.`
          : graphPassed
            ? "Every returned edge points to a node present in the graph."
            : "The graph endpoint responded, but returned no nodes to validate.",
        details: [
          `Nodes: ${graph.nodes.length}`,
          `Edges: ${graph.edges.length}`,
          `Dangling edges: ${danglingEdges.length}`,
          `Graph truncated by API: ${graph.truncated ? "yes" : "no"}`,
        ],
        durationMs: Math.round(performance.now() - started),
      });

      return { scenarios, relationships };
    } catch (error) {
      const message = error instanceof Error ? error.message : "Dataset checks failed.";
      setCheck({
        id: "dataset",
        title: "Dataset consistency",
        status: "failed",
        summary: message,
        details: [],
        durationMs: Math.round(performance.now() - started),
      });
      setCheck({
        id: "graph",
        title: "Knowledge graph integrity",
        status: "failed",
        summary: message,
        details: [],
        durationMs: Math.round(performance.now() - started),
      });
      return null;
    }
  };

  const runStrategyComparison = async (
    context?: Awaited<ReturnType<typeof runDatasetChecks>>,
  ) => {
    setStrategyResults([]);
    if (!activeDatasetId) {
      setRunError("Select an imported dataset before running recovery experiments.");
      return;
    }

    const data = context ?? (await runDatasetChecks());
    if (!data) {
      setRunError("Could not load dataset context for the recovery comparison.");
      return;
    }

    // Relationship rows may not include scenario_id; resolve it independently.
    const relationship = data.relationships.find(
      (item) => item.subject?.trim() && item.predicate?.trim(),
    );
    const scenarioId =
      relationship?.scenario_id?.trim() ||
      data.scenarios.find((item) => item.scenario_id?.trim())?.scenario_id?.trim();
    const knownEndpoint = relationship?.subject?.trim();
    const predicate = relationship?.predicate?.trim();

    if (!scenarioId || !knownEndpoint || !predicate) {
      setRunError(
        "Recovery comparison needs at least one relationship with a subject and predicate, plus at least one scenario with an ID. Check the imported dataset mappings.",
      );
      return;
    }

    setRunError(null);
    const strategies = ["graph", "llm", "hybrid"] as const;
    const results = await Promise.all(
      strategies.map(async (strategy): Promise<StrategyResult> => {
        const started = performance.now();
        try {
          const response = await api.recover(activeDatasetId, {
            scenario_id: scenarioId,
            predicate,
            known_endpoint: knownEndpoint,
            missing_endpoint: "object",
            limit: 20,
            strategy,
          });
          return {
            strategy,
            status: "passed",
            durationMs: Math.round(performance.now() - started),
            candidateCount: response.recovery?.candidates?.length ?? 0,
            response,
          };
        } catch (error) {
          return {
            strategy,
            status: "failed",
            durationMs: Math.round(performance.now() - started),
            candidateCount: 0,
            response: null,
            error: error instanceof Error ? error.message : "Recovery request failed.",
          };
        }
      }),
    );
    setStrategyResults(results);
  };

  const runAllExperiments = async () => {
    if (running) return;
    setRunning(true);
    setRunError(null);
    setLastRun(null);
    setStrategyResults([]);
    try {
      const healthy = await runConnectivity();
      const data = await runDatasetChecks();
      if (healthy && data) {
        await runStrategyComparison(data);
      } else if (!healthy) {
        setRunError("API health check failed. Dataset checks were still attempted; recovery comparison was skipped.");
      } else {
        setRunError("Dataset checks did not complete successfully; recovery comparison was skipped.");
      }
      setLastRun(new Date().toLocaleString());
    } finally {
      setRunning(false);
    }
  };

  const passedCount = checks.filter((check) => check.status === "passed").length;
  const failedCount = checks.filter((check) => check.status === "failed").length;

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Validation / 05"
        title="Experiments"
        description="Run repeatable, one-click checks against the live API, imported HAZOP data, graph structure, and recovery strategies."
        actions={<Badge variant="outline">{activeDatasetId ?? "No dataset selected"}</Badge>}
      />

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <FlaskConical className="size-4 text-muted-foreground" />
                <CardTitle>Experiment runner</CardTitle>
              </div>
              <CardDescription>
                The full suite calls the configured backend. Results reflect actual responses, not simulated metrics.
              </CardDescription>
            </div>
            <Button onClick={() => void runAllExperiments()} disabled={running}>
              {running ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
              {running ? "Running suite…" : "Run all experiments"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="border border-border p-3">
              <p className="text-[10px] text-muted-foreground">Checks passed</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">{passedCount}</p>
            </div>
            <div className="border border-border p-3">
              <p className="text-[10px] text-muted-foreground">Checks failed</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">{failedCount}</p>
            </div>
            <div className="border border-border p-3">
              <p className="text-[10px] text-muted-foreground">Last run</p>
              <p className="mt-2 text-xs font-medium">{lastRun ?? "Not run yet"}</p>
            </div>
          </div>

          <div className="flex flex-wrap gap-2 border-t border-border pt-4">
            <Button variant="outline" size="sm" disabled={running} onClick={() => void runConnectivity()}>
              <Activity className="size-3.5" /> Test API
            </Button>
            <Button variant="outline" size="sm" disabled={running} onClick={() => void runDatasetChecks()}>
              <ShieldCheck className="size-3.5" /> Validate dataset and graph
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={running || !activeDatasetId}
              onClick={() => void runStrategyComparison()}
            >
              <GitBranch className="size-3.5" /> Compare recovery strategies
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={running}
              onClick={() => {
                setChecks((current) => current.map((item) => ({ ...item, status: "idle", details: [], durationMs: undefined })));
                setStrategyResults([]);
                setRunError(null);
                setLastRun(null);
              }}
            >
              <RotateCcw className="size-3.5" /> Reset results
            </Button>
          </div>
          {runError && <p className="text-xs text-destructive" role="alert">{runError}</p>}
        </CardContent>
      </Card>

      <section className="space-y-3">
        <div>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2"><FlaskConical className="size-4 text-muted-foreground" /><CardTitle>Research benchmark</CardTitle></div>
              <CardDescription>Runs the repository benchmark against held-out scenarios and displays real metrics and per-case predictions. Scores reflect silver row associations, not expert-verified causal truth.</CardDescription>
            </div>
            <Button onClick={() => void runResearchBenchmark()} disabled={benchmarkRunning}>
              {benchmarkRunning ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
              {benchmarkRunning ? "Running benchmark…" : "Run benchmark"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-4">
            <label className="space-y-1 text-xs"><span className="text-muted-foreground">Dataset track</span>
              <select value={benchmarkDataset} onChange={(event) => setBenchmarkDataset(event.target.value as BenchmarkRun["dataset"])} disabled={benchmarkRunning} className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm">
                <option value="t301-reconstructed">T-301 reconstructed</option><option value="t301-synthetic">T-301 synthetic continuation</option><option value="dwsim-synthetic">DWSIM synthetic</option>
              </select>
            </label>
            <label className="space-y-1 text-xs"><span className="text-muted-foreground">Random seed</span><input type="number" min={0} max={1000000} value={benchmarkSeed} onChange={(event) => setBenchmarkSeed(Number(event.target.value))} disabled={benchmarkRunning} className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm" /></label>
            <label className="space-y-1 text-xs"><span className="text-muted-foreground">Test split (%)</span><input type="number" min={5} max={50} step={5} value={Math.round(benchmarkTestFraction * 100)} onChange={(event) => setBenchmarkTestFraction(Number(event.target.value) / 100)} disabled={benchmarkRunning} className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm" /></label>
            <label className="flex items-center gap-2 self-end pb-2 text-sm"><input type="checkbox" checked={benchmarkWithLlm} onChange={(event) => setBenchmarkWithLlm(event.target.checked)} disabled={benchmarkRunning} />Include LLM methods</label>
          </div>
          {benchmarkError && <p className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive" role="alert">{benchmarkError}</p>}
          {benchmarkResult && <div className="space-y-4 border-t border-border pt-4">
            <div className="flex flex-wrap items-center justify-between gap-2"><div><p className="text-sm font-medium">Run completed · {benchmarkResult.dataset}</p><p className="text-xs text-muted-foreground">Seed {benchmarkResult.seed} · {Math.round(benchmarkResult.test_fraction * 100)}% test split · {benchmarkResult.cases.length} test cases</p></div>
              <div className="flex flex-wrap gap-2"><Button variant="outline" size="sm" onClick={() => downloadBenchmarkArtifact("hazop-benchmark-report.md", benchmarkResult.artifacts.report_md, "text/markdown")}>Download report</Button><Button variant="outline" size="sm" onClick={() => downloadBenchmarkArtifact("hazop-benchmark-summary.csv", benchmarkResult.artifacts.summary_csv, "text/csv")}>Download summary CSV</Button><Button variant="outline" size="sm" onClick={() => downloadBenchmarkArtifact("hazop-benchmark-predictions.csv", benchmarkResult.artifacts.predictions_csv, "text/csv")}>Download predictions CSV</Button></div>
            </div>
            <div className="overflow-x-auto rounded-md border border-border"><table className="w-full min-w-[760px] text-left text-xs"><thead className="bg-muted/50 text-muted-foreground"><tr><th className="p-3 font-medium">Method</th><th className="p-3 font-medium">Status</th><th className="p-3 font-medium">Evaluated</th><th className="p-3 font-medium">Micro F1</th><th className="p-3 font-medium">Precision</th><th className="p-3 font-medium">Recall</th><th className="p-3 font-medium">Coverage</th><th className="p-3 font-medium">Abstention</th><th className="p-3 font-medium">Latency</th></tr></thead>
              <tbody>{Object.entries(benchmarkResult.method_summary).map(([method, metrics]) => <tr key={method} className="border-t border-border"><td className="p-3 font-medium">{method}</td><td className="p-3"><Badge variant={metrics.status === "completed" || metrics.status === "ok" ? "secondary" : "outline"}>{metrics.status ?? "reported"}</Badge></td><td className="p-3 tabular-nums">{metrics.n_evaluated ?? 0} / {metrics.n_total_test_cases ?? benchmarkResult.cases.length}</td><td className="p-3 tabular-nums">{formatBenchmarkMetric(metrics.micro_f1)}</td><td className="p-3 tabular-nums">{formatBenchmarkMetric(metrics.micro_precision)}</td><td className="p-3 tabular-nums">{formatBenchmarkMetric(metrics.micro_recall)}</td><td className="p-3 tabular-nums">{formatBenchmarkMetric(metrics.coverage)}</td><td className="p-3 tabular-nums">{formatBenchmarkMetric(metrics.abstention_rate)}</td><td className="p-3 tabular-nums">{metrics.mean_latency_ms == null ? "—" : metrics.mean_latency_ms.toFixed(1) + " ms"}</td></tr>)}</tbody></table></div>
            <div className="space-y-2"><h3 className="text-sm font-semibold">Per-case predictions</h3><p className="text-xs text-muted-foreground">Expand a test case to inspect its silver target and each method’s predictions. Targets are derived from co-listed row fields.</p>
              <div className="space-y-2">{benchmarkResult.cases.map((item) => <details key={item.case_id} className="rounded-md border border-border px-3 py-2"><summary className="cursor-pointer text-xs font-medium">{item.case_id} · {item.known_endpoint}</summary><div className="mt-3 space-y-2 text-xs"><p><span className="text-muted-foreground">Silver target:</span> {item.gold_values.join(" · ") || "None"}</p>
                {Object.entries(item.methods).map(([method, result]) => <div key={method} className="rounded-md bg-muted/30 p-2"><p className="font-medium">{method} · {result.status ?? "unknown"} · {result.latency_ms == null ? "latency unavailable" : result.latency_ms.toFixed(1) + " ms"}</p><p className="mt-1 text-muted-foreground">Predictions: {(result.predictions ?? []).map((prediction) => prediction.value).join(" · ") || "No candidates / abstained"}</p>{(result.predictions ?? []).some((prediction) => prediction.evidence) && <ul className="mt-1 list-disc pl-4 text-muted-foreground">{(result.predictions ?? []).filter((prediction) => prediction.evidence).map((prediction, index) => <li key={method + "-" + index}>{prediction.evidence}</li>)}</ul>}</div>)}
              </div></details>)}</div>
            </div>
          </div>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Automated benchmark suite</CardTitle>
          <CardDescription>Run a reproducible dataset × seed matrix and persist every run. Scores use silver row associations, not expert-verified causal truth.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            {([
              ["t301-reconstructed", "T-301 reconstructed"],
              ["t301-synthetic", "T-301 synthetic"],
              ["dwsim-synthetic", "DWSIM synthetic"],
            ] as const).map(([value, label]) => (
              <label key={value} className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={suiteDatasets.includes(value)} disabled={suiteRunning}
                  onChange={(event) => setSuiteDatasets((current) => event.target.checked ? [...current, value] : current.filter((item) => item !== value))} />
                {label}
              </label>
            ))}
          </div>
          <label className="block space-y-1 text-xs"><span className="text-muted-foreground">Random seeds (comma-separated; up to 10)</span>
            <input value={suiteSeeds} onChange={(event) => setSuiteSeeds(event.target.value)} disabled={suiteRunning} className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm" />
          </label>
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={() => void runBenchmarkSuite()} disabled={suiteRunning || !suiteDatasets.length}>
              {suiteRunning ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
              {suiteRunning ? "Running suite…" : "Run benchmark suite"}
            </Button>
            <span className="text-xs text-muted-foreground">{suiteDatasets.length * suiteSeeds.split(",").filter((item) => item.trim()).length} planned runs</span>
          </div>
          {suiteError && <p role="alert" className="text-sm text-destructive">{suiteError}</p>}
          {suiteResult && <>
            <div className="rounded-md border border-border p-3 text-xs"><p className="font-medium">Run {suiteResult.run_id}</p><p className="mt-1 text-muted-foreground">{suiteResult.runs.length} completed · {suiteResult.failures.length} failed</p><p className="mt-2 text-muted-foreground">{suiteResult.label_warning}</p></div>
            <div className="overflow-x-auto rounded-md border border-border"><table className="w-full min-w-[680px] text-left text-xs">
              <thead className="bg-muted/50 text-muted-foreground"><tr><th className="p-2">Dataset</th><th className="p-2">Seed</th><th className="p-2">Cases</th><th className="p-2">Method</th><th className="p-2">Micro F1</th><th className="p-2">Coverage</th></tr></thead>
              <tbody>{suiteResult.runs.flatMap((run) => Object.entries(run.method_summary).map(([method, metrics]) => <tr key={run.dataset + run.seed + method} className="border-t border-border"><td className="p-2">{run.dataset}</td><td className="p-2">{run.seed}</td><td className="p-2">{run.case_count}</td><td className="p-2">{method} · {metrics.status ?? "unknown"}</td><td className="p-2">{formatBenchmarkMetric(metrics.micro_f1)}</td><td className="p-2">{formatBenchmarkMetric(metrics.coverage)}</td></tr>))}</tbody>
            </table></div>
            {suiteResult.failures.map((failure) => <p key={failure.dataset + failure.seed} className="text-xs text-destructive">{failure.dataset} · seed {failure.seed}: {failure.error}</p>)}
          </>}
          {savedRuns.length > 0 && <div className="space-y-1 border-t border-border pt-3"><h3 className="text-sm font-medium">Saved suite runs</h3>{savedRuns.slice(0, 5).map((run) => <p key={run.run_id} className="text-xs text-muted-foreground">{run.run_id} · {run.runs.length} completed · {run.failures.length} failed</p>)}</div>}
        </CardContent>
      </Card>

      <section className="space-y-3">
        <div>
          <h2 className="text-sm font-semibold">System and data checks</h2>
          <p className="mt-1 text-[11px] text-muted-foreground">Each check can run independently or as part of the full suite.</p>
        </div>
        <div className="grid gap-3 lg:grid-cols-3">
          {checks.filter((check) => ["api", "dataset", "graph"].includes(check.id)).map((check) => (
            <Card key={check.id}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="text-sm">{check.title}</CardTitle>
                  <StatusBadge status={check.status} />
                </div>
                <CardDescription>{check.summary}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                  <ResultIcon status={check.status} />
                  {check.durationMs !== undefined ? <><Clock3 className="size-3" /> {check.durationMs} ms</> : "Waiting for run"}
                </div>
                {check.details.length > 0 && (
                  <ul className="space-y-1 border-t border-border pt-3">
                    {check.details.map((detail) => <li key={detail} className="break-words text-[10px] leading-5 text-muted-foreground">{detail}</li>)}
                  </ul>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Workflow className="size-4 text-muted-foreground" />
            <CardTitle>Recovery strategy comparison</CardTitle>
          </div>
          <CardDescription>
            The runner selects a relationship from the active dataset and sends the same scenario, predicate, and known subject to each available strategy. Candidate counts and response times are reported; these are not precision/recall scores.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {strategyResults.length === 0 ? (
            <div className="flex min-h-36 flex-col items-center justify-center gap-2 border border-dashed border-border bg-muted/20 p-6 text-center">
              <GitBranch className="size-6 text-muted-foreground" />
              <p className="text-xs font-medium">No recovery comparison run</p>
              <p className="max-w-lg text-[11px] leading-5 text-muted-foreground">
                Choose Compare recovery strategies or Run all experiments. The dataset must contain at least one usable relationship.
              </p>
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-3">
              {strategyResults.map((result) => (
                <div key={result.strategy} className="space-y-3 border border-border p-4">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-semibold">{strategyLabels[result.strategy]}</p>
                    <Badge variant={result.status === "passed" ? "secondary" : "destructive"}>
                      {result.status === "passed" ? "Responded" : "Failed"}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div><p className="text-[10px] text-muted-foreground">Candidates</p><p className="mt-1 text-xl font-semibold tabular-nums">{result.candidateCount}</p></div>
                    <div><p className="text-[10px] text-muted-foreground">Latency</p><p className="mt-1 text-xl font-semibold tabular-nums">{result.durationMs}<span className="ml-1 text-[10px] font-normal">ms</span></p></div>
                  </div>
                  {result.error ? (
                    <p className="break-words border-t border-border pt-3 text-[10px] leading-5 text-destructive">{result.error}</p>
                  ) : (
                    <div className="border-t border-border pt-3 text-[10px] leading-5 text-muted-foreground">
                      Validation: {result.response?.validation.valid ? "passed" : "failed"}
                      {result.response?.validation.errors.length ? ` · ${result.response.validation.errors.length} validation error(s)` : ""}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
