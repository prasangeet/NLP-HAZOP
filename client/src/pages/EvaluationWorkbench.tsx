import { useMemo, useState } from "react";
import { Download, ChevronLeft, ChevronRight, Search, Play, LoaderCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import PageHeader from "@/components/layout/PageHeader";
import { api, type BenchmarkRun, type BenchmarkCaseResult } from "@/lib/api";

const METHODS = [
  ["graph", "Graph"],
  ["tfidf", "TF-IDF"],
  ["graph_tfidf", "Graph + TF-IDF"],
  ["prompt_only_llm", "Prompt-only LLM"],
  ["hybrid_graph_vector_llm", "Hybrid"],
] as const;

function downloadJson(run: BenchmarkRun) {
  const blob = new Blob([JSON.stringify(run, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `hazop-${run.dataset}-seed-${run.seed}-cases.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function predictionValues(item: BenchmarkCaseResult["methods"][string] | undefined) {
  return item?.predictions ?? [];
}

export default function EvaluationWorkbench() {
  const [dataset, setDataset] = useState("t301-reconstructed");
  const [seed, setSeed] = useState("42");
  const [fraction, setFraction] = useState("0.2");
  const [withLlm, setWithLlm] = useState(true);
  const [llmCap, setLlmCap] = useState("15");
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [methodFilter, setMethodFilter] = useState("all");
  const [caseIndex, setCaseIndex] = useState(0);

  const filteredCases = useMemo(() => {
    if (!run) return [];
    const normalized = query.trim().toLowerCase();
    return run.cases.filter((item) => {
      const matchesText = !normalized || [item.case_id, item.scenario_id, item.known_endpoint, ...item.gold_values]
        .some((value) => value.toLowerCase().includes(normalized));
      const matchesMethod = methodFilter === "all" || Boolean(item.methods[methodFilter]);
      return matchesText && matchesMethod;
    });
  }, [run, query, methodFilter]);

  const selected = filteredCases[Math.min(caseIndex, Math.max(0, filteredCases.length - 1))];

  const execute = async () => {
    setRunning(true);
    setError(null);
    setRun(null);
    setCaseIndex(0);
    try {
      const result = await api.runBenchmark({
        dataset,
        seed: Number(seed),
        test_fraction: Number(fraction),
        with_llm: withLlm,
        max_llm_cases: withLlm ? Number(llmCap) : 0,
        cross_validation_folds: 0,
      });
      setRun(result);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Benchmark execution failed.");
    } finally {
      setRunning(false);
    }
  };

  const caseStatus = (item: BenchmarkCaseResult, method: string) => {
    const result = item.methods[method];
    if (!result) return "Not returned";
    if (result.status?.startsWith("not_evaluated")) return "Not evaluated";
    const predicted = predictionValues(result).map((value) => value.value);
    const gold = new Set(item.gold_values.map((value) => value.toLowerCase()));
    const hits = predicted.filter((value) => gold.has(value.toLowerCase())).length;
    return hits ? "Overlap" : predicted.length ? "No overlap" : "Abstained";
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Evaluation Workbench" description="Run a reproducible benchmark, then inspect each held-out case and compare method outputs." />
      <Card>
        <CardHeader><CardTitle className="text-base">Benchmark configuration</CardTitle><CardDescription>Each run evaluates the full held-out case set for graph and retrieval methods. The LLM cap limits only LLM-dependent methods.</CardDescription></CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
          <label className="space-y-1 text-xs">Dataset<Select value={dataset} onValueChange={setDataset}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="t301-reconstructed">T-301 reconstructed</SelectItem><SelectItem value="t301-synthetic">T-301 synthetic</SelectItem><SelectItem value="dwsim-synthetic">DWSIM synthetic</SelectItem></SelectContent></Select></label>
          <label className="space-y-1 text-xs">Seed<Input value={seed} onChange={(event) => setSeed(event.target.value)} inputMode="numeric" /></label>
          <label className="space-y-1 text-xs">Test fraction<Select value={fraction} onValueChange={setFraction}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{["0.1","0.2","0.3","0.4","0.5"].map((value) => <SelectItem key={value} value={value}>{Math.round(Number(value)*100)}%</SelectItem>)}</SelectContent></Select></label>
          <label className="space-y-1 text-xs">LLM evaluation cap<Select value={llmCap} onValueChange={setLlmCap}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{["15","30","50","100"].map((value) => <SelectItem key={value} value={value}>{value} cases (maximum)</SelectItem>)}</SelectContent></Select></label>
          <div className="flex flex-col justify-end gap-2"><label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={withLlm} onChange={(event) => setWithLlm(event.target.checked)} /> Include LLM methods</label><Button onClick={execute} disabled={running || !/^\d+$/.test(seed) || Number(seed)>1000000}><Play className="mr-2 size-4" />{running ? "Running benchmark…" : "Run evaluation"}</Button></div>
        </CardContent>
      </Card>
      {running && <div className="flex items-center gap-2 text-sm text-muted-foreground"><LoaderCircle className="size-4 animate-spin" />Benchmark runs synchronously; case-level progress and pause/resume are not exposed by the current API.</div>}
      {error && <p className="text-sm text-destructive">{error}</p>}
      {run && <>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Card><CardContent className="p-4"><p className="text-xs text-muted-foreground">Held-out cases</p><p className="text-2xl font-semibold">{run.cases.length}</p></CardContent></Card>
          <Card><CardContent className="p-4"><p className="text-xs text-muted-foreground">Dataset / seed</p><p className="text-lg font-semibold">{run.dataset} / {run.seed}</p></CardContent></Card>
          <Card><CardContent className="p-4"><p className="text-xs text-muted-foreground">LLM evaluated</p><p className="text-2xl font-semibold">{String(run.metadata.llm_enabled ? run.metadata.llm_max_cases ?? "enabled" : "disabled")}</p></CardContent></Card>
          <Card><CardContent className="flex h-full items-center justify-between p-4"><div><p className="text-xs text-muted-foreground">Export</p><p className="text-sm font-medium">Complete run data</p></div><Button size="sm" variant="outline" onClick={() => downloadJson(run)}><Download className="mr-2 size-4" />JSON</Button></CardContent></Card>
        </div>
        <Card>
          <CardHeader><div className="flex flex-wrap items-center justify-between gap-2"><div><CardTitle className="text-base">Case review</CardTitle><CardDescription>Inspect gold row associations, predictions, provenance, evidence, and latency.</CardDescription></div><Badge variant="outline">{filteredCases.length} / {run.cases.length} cases</Badge></div></CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-2 md:grid-cols-[1fr_220px]"><div className="relative"><Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" /><Input className="pl-9" placeholder="Search case, scenario, endpoint, or target…" value={query} onChange={(event) => { setQuery(event.target.value); setCaseIndex(0); }} /></div><Select value={methodFilter} onValueChange={(value) => { setMethodFilter(value); setCaseIndex(0); }}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All methods</SelectItem>{METHODS.map(([key,label]) => <SelectItem key={key} value={key}>{label}</SelectItem>)}</SelectContent></Select></div>
            {!selected ? <p className="py-8 text-center text-sm text-muted-foreground">No cases match these filters.</p> : <>
              <div className="flex items-center justify-between border-y py-3"><div><p className="font-mono text-sm font-semibold">{selected.case_id}</p><p className="text-xs text-muted-foreground">Scenario {selected.scenario_id} · {selected.record_type}</p></div><div className="flex items-center gap-2"><Button size="icon" variant="outline" disabled={caseIndex===0} onClick={() => setCaseIndex((value)=>Math.max(0,value-1))}><ChevronLeft className="size-4" /></Button><span className="text-xs tabular-nums">{caseIndex+1} / {filteredCases.length}</span><Button size="icon" variant="outline" disabled={caseIndex>=filteredCases.length-1} onClick={() => setCaseIndex((value)=>Math.min(filteredCases.length-1,value+1))}><ChevronRight className="size-4" /></Button></div></div>
              <div className="grid gap-3 lg:grid-cols-2"><div className="rounded-md border p-3"><p className="text-xs text-muted-foreground">Known endpoint</p><p className="mt-1 font-medium">{selected.known_endpoint}</p><p className="mt-4 text-xs text-muted-foreground">Silver row-association targets</p><div className="mt-2 flex flex-wrap gap-1">{selected.gold_values.map((value)=> <Badge key={value} variant="secondary">{value}</Badge>)}</div></div><div className="rounded-md border p-3"><p className="text-xs text-muted-foreground">Case metadata</p><p className="mt-1 text-sm">Fold: {selected.fold ?? "Holdout"}</p><p className="text-sm">Methods returned: {Object.keys(selected.methods).length}</p><p className="text-sm">Methods with predictions: {Object.values(selected.methods).filter((value)=>predictionValues(value).length>0).length}</p></div></div>
              <div className="grid gap-3 xl:grid-cols-2">{METHODS.map(([key,label])=>{const result=selected.methods[key];const predictions=predictionValues(result);return <div key={key} className="rounded-md border p-3"><div className="flex items-center justify-between gap-2"><p className="font-medium">{label}</p><Badge variant="outline">{caseStatus(selected,key)}</Badge></div><p className="mt-1 text-xs text-muted-foreground">{result?.status ?? "No result"}{typeof result?.latency_ms==="number" ? ` · ${result.latency_ms.toFixed(1)} ms` : ""}</p>{predictions.length ? <div className="mt-3 space-y-2">{predictions.map((prediction,index)=><div key={prediction.value+index} className="rounded bg-muted/40 p-2"><div className="flex flex-wrap items-center gap-2"><span className="text-sm font-medium">{prediction.value}</span><Badge variant="secondary">{prediction.origin ?? "origin unavailable"}</Badge></div>{prediction.evidence && <p className="mt-1 text-xs text-muted-foreground">{prediction.evidence}</p>}</div>)}</div> : <p className="mt-3 text-xs text-muted-foreground">No prediction returned.</p>}</div>})}</div>
            </>}
          </CardContent>
        </Card>
        <p className="text-xs text-muted-foreground">Evaluation uses silver row-association labels. It does not establish causal validity or safety certification.</p>
      </>}
    </div>
  );
}
