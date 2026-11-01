import { useEffect, useMemo, useState } from "react";
import { Download, FlaskConical, LoaderCircle, RefreshCw } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import PageHeader from "@/components/layout/PageHeader";
import { api, type BenchmarkSuiteRun } from "@/lib/api";

type Props = { runId: string | null; onOpenRun: (runId: string) => void };
type StoredResults = {
  cases?: Array<{
    case_id: string; scenario_id: string; known_endpoint: string; gold_values: string[];
    methods?: Record<string, { status?: string; predictions?: Array<{ value: string; evidence?: string | null; origin?: string }> }>;
  }>;
  metadata?: Record<string, unknown>;
  method_summary?: Record<string, Record<string, unknown>>;
  matched_case_summary?: Record<string, Record<string, unknown>>;
  fold_results?: Array<Record<string, unknown>>;
};

function metric(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—";
}

function saveText(filename: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function ExperimentResults({ runId, onOpenRun }: Props) {
  const [runs, setRuns] = useState<BenchmarkSuiteRun[]>([]);
  const [selected, setSelected] = useState<BenchmarkSuiteRun | null>(null);
  const [details, setDetails] = useState<Array<{ key: string; dataset: string; seed: number; hops: number; data: StoredResults }>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reportDownloading, setReportDownloading] = useState(false);

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const response = await api.listBenchmarkRuns();
      setRuns(response.runs);
      const match = response.runs.find((run) => run.run_id === runId) ?? null;
      setSelected(match);
      if (!match) { setDetails([]); return; }
      const artifacts = await Promise.all(match.runs.map(async (run) => {
        const artifact = await api.getBenchmarkArtifact(match.run_id, run.dataset, run.seed, run.max_hops, "results.json");
        return { key: run.dataset + "-" + run.seed + "-" + run.max_hops, dataset: run.dataset, seed: run.seed, hops: run.max_hops, data: JSON.parse(artifact.content) as StoredResults };
      }));
      setDetails(artifacts);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load experiment results.");
    } finally { setLoading(false); }
  };

  useEffect(() => { void load(); }, [runId]);

  const downloadFullReport = async () => {
    if (!selected) return;
    setReportDownloading(true);
    setError(null);
    try {
      const report = await api.getBenchmarkFullReport(selected.run_id);
      if (!report.content.trim()) throw new Error("The server returned an empty full report.");
      saveText(report.filename || "nlp-hazop-" + selected.run_id + "-full-report.md", report.content, "text/markdown;charset=utf-8");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not download the full experiment report.");
    } finally {
      setReportDownloading(false);
    }
  };

  const aggregateRows = useMemo(() => selected ? Object.values(selected.aggregate_summary) : [], [selected]);

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader eyebrow="Research space / Results" title="Experiment results" description="Browse persisted benchmark runs, inspect per-method metrics and held-out cases, and retrieve original artifacts." actions={<Button size="sm" variant="outline" onClick={() => void load()} disabled={loading}><RefreshCw className={loading ? "size-3.5 animate-spin" : "size-3.5"} />Refresh</Button>} />
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      {loading ? <div className="flex items-center gap-2 py-12 text-sm text-muted-foreground"><LoaderCircle className="size-4 animate-spin" />Loading saved experiment results…</div> : (
        <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
          <Card className="h-fit">
            <CardHeader><CardTitle className="text-sm">Saved experiments</CardTitle><CardDescription>{runs.length} persisted suite runs</CardDescription></CardHeader>
            <CardContent className="space-y-2">
              {runs.length === 0 && <p className="text-xs text-muted-foreground">No saved experiment runs yet.</p>}
              {runs.map((run) => <button key={run.run_id} onClick={() => onOpenRun(run.run_id)} className={`w-full border p-3 text-left hover:bg-muted/40 ${run.run_id === runId ? "border-primary bg-muted/30" : "border-border"}`}>
                <p className="break-all text-xs font-medium">{run.run_id}</p><p className="mt-1 text-[10px] text-muted-foreground">{run.runs.length} completed · {run.failures.length} failed</p><p className="mt-1 text-[10px] text-muted-foreground">{run.datasets.join(", ")}</p>
              </button>)}
            </CardContent>
          </Card>
          <div className="min-w-0 space-y-4">
            {!selected ? <Card><CardContent className="flex min-h-48 flex-col items-center justify-center gap-2 text-center"><FlaskConical className="size-7 text-muted-foreground" /><p className="text-sm font-medium">Select an experiment run</p><p className="text-xs text-muted-foreground">Choose a saved run to inspect its metrics, folds, predictions, and artifacts.</p></CardContent></Card> : <>
              <Card>
                <CardHeader><div className="flex flex-wrap items-center justify-between gap-2"><div className="min-w-0"><CardTitle className="break-all text-sm">{selected.run_id}</CardTitle><CardDescription>{new Date(selected.created_at).toLocaleString()} · {selected.runs.length} completed · {selected.failures.length} failed</CardDescription></div><div className="flex flex-wrap items-center gap-2"><Button size="sm" onClick={() => void downloadFullReport()} disabled={reportDownloading}><Download className={reportDownloading ? "size-3.5 animate-pulse" : "size-3.5"} />{reportDownloading ? "Preparing report…" : "Download full report"}</Button><Badge variant="outline">{selected.label_status}</Badge></div></div></CardHeader>
                <CardContent className="space-y-2 text-xs"><p>{selected.label_warning}</p><p className="text-muted-foreground">Datasets: {selected.datasets.join(", ")} · Seeds: {selected.seeds.join(", ")} · Graph hops: {selected.graph_hops.join(", ")}</p></CardContent>
              </Card>
              <Card><CardHeader><CardTitle className="text-sm">Across-seed metrics</CardTitle><CardDescription>Full-test and matched-case summaries remain separate.</CardDescription></CardHeader><CardContent className="overflow-x-auto"><table className="w-full min-w-[650px] text-left text-xs"><thead className="bg-muted/50"><tr>{["Scope","Dataset","Hops","Method","Seeds","Mean F1","95% CI","Coverage"].map((h) => <th key={h} className="p-2 font-medium">{h}</th>)}</tr></thead><tbody>{aggregateRows.map((item) => { const f1=item.metrics.micro_f1; const coverage=item.metrics.coverage; return <tr key={item.scope+item.dataset+item.max_hops+item.method} className="border-t border-border"><td className="p-2">{item.scope === "matched_cases" ? "Matched" : "Full test"}</td><td className="p-2">{item.dataset}</td><td className="p-2">{item.max_hops}</td><td className="p-2">{item.method}</td><td className="p-2">{f1?.n_seeds ?? 0}</td><td className="p-2">{f1 ? metric(f1.mean) : "—"}</td><td className="p-2">{f1?.ci95_low == null || f1.ci95_high == null ? "Not estimable" : metric(f1.ci95_low)+"–"+metric(f1.ci95_high)}</td><td className="p-2">{coverage ? metric(coverage.mean) : "—"}</td></tr>; })}</tbody></table></CardContent></Card>
              {details.map((item) => <Card key={item.key}>
                <CardHeader><div className="flex flex-wrap items-center justify-between gap-2"><div><CardTitle className="text-sm">{item.dataset} · seed {item.seed} · {item.hops} hops</CardTitle><CardDescription>{item.data.cases?.length ?? 0} evaluated cases</CardDescription></div><div className="flex flex-wrap gap-2"><Button size="sm" variant="outline" onClick={async()=>saveText(item.key+"-results.json",(await api.getBenchmarkArtifact(selected.run_id,item.dataset,item.seed,item.hops,"results.json")).content,"application/json")}><Download className="size-3.5"/>JSON</Button><Button size="sm" variant="outline" onClick={async()=>saveText(item.key+"-report.md",(await api.getBenchmarkArtifact(selected.run_id,item.dataset,item.seed,item.hops,"report.md")).content,"text/markdown")}>Report</Button><Button size="sm" variant="outline" onClick={async()=>saveText(item.key+"-predictions.csv",(await api.getBenchmarkArtifact(selected.run_id,item.dataset,item.seed,item.hops,"predictions.csv")).content,"text/csv")}>Predictions CSV</Button></div></div></CardHeader>
                <CardContent className="space-y-3">
                  <div className="overflow-x-auto"><table className="w-full min-w-[580px] text-left text-xs"><thead className="bg-muted/50"><tr>{["Method","Status","Cases","Micro F1","Coverage","Latency"].map(h=><th key={h} className="p-2">{h}</th>)}</tr></thead><tbody>{Object.entries(item.data.method_summary ?? {}).map(([method,m])=><tr key={method} className="border-t border-border"><td className="p-2 font-medium">{method}</td><td className="p-2">{String(m.status ?? "—")}</td><td className="p-2">{String(m.n_evaluated ?? "—")}</td><td className="p-2">{metric(m.micro_f1)}</td><td className="p-2">{metric(m.coverage)}</td><td className="p-2">{typeof m.mean_latency_ms === "number" ? m.mean_latency_ms.toFixed(1)+" ms" : "—"}</td></tr>)}</tbody></table></div>
                  <details className="border-t border-border pt-3"><summary className="cursor-pointer text-xs font-medium">Inspect per-case predictions ({item.data.cases?.length ?? 0})</summary><div className="mt-3 max-h-[560px] space-y-2 overflow-auto">{(item.data.cases ?? []).map((c)=><details key={c.case_id} className="border border-border p-3"><summary className="cursor-pointer text-xs">{c.case_id} · {c.known_endpoint}</summary><div className="mt-2 space-y-2 text-xs"><p><span className="text-muted-foreground">Silver targets:</span> {c.gold_values.join(" · ") || "None"}</p>{Object.entries(c.methods ?? {}).map(([method,result])=><div key={method} className="bg-muted/30 p-2"><p className="font-medium">{method} · {result.status ?? "unknown"}</p><p className="mt-1 text-muted-foreground">Predictions: {(result.predictions ?? []).map(p=>p.value).join(" · ") || "No candidates / abstained"}</p>{(result.predictions ?? []).filter(p=>p.evidence).map((p,i)=><p key={i} className="mt-1 text-muted-foreground">Evidence: {p.evidence}</p>)}</div>)}</div></details>)}</div></details>
                </CardContent>
              </Card>)}
              {selected.failures.map((failure,i)=><p key={i} className="text-xs text-destructive">{failure.dataset} · seed {failure.seed} · {failure.max_hops} hops: {failure.error}</p>)}
            </>}
          </div>
        </div>
      )}
    </div>
  );
}
