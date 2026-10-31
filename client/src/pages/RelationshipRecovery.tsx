import { useState } from "react";
import { BrainCircuit, GitBranch, LoaderCircle, Search, Workflow, ShieldAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import PageHeader from "@/components/layout/PageHeader";
import { api, type RecoveryResponse } from "@/lib/api";
import { useWorkspaceStore } from "@/lib/workspace-store";

const strategies = [
  { id: "graph", title: "Graph-based", description: "Use graph and multi-hop evidence.", icon: GitBranch },
  { id: "llm", title: "LLM-based", description: "Use language-model reasoning.", icon: BrainCircuit },
  { id: "hybrid", title: "Hybrid", description: "Combine graph evidence and language-model reasoning.", icon: Workflow },
] as const;

type RecoveryStrategy = (typeof strategies)[number]["id"];

export default function RelationshipRecovery() {
  const activeDatasetId = useWorkspaceStore((state) => state.activeDatasetId);
  const [strategy, setStrategy] = useState<RecoveryStrategy>("hybrid");
  const [scenarioId, setScenarioId] = useState("");
  const [predicate, setPredicate] = useState("leads_to_consequence");
  const [knownEndpoint, setKnownEndpoint] = useState("");
  const [missingEndpoint, setMissingEndpoint] = useState<"subject" | "object">("object");
  const [result, setResult] = useState<RecoveryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runRecovery = async () => {
    if (!activeDatasetId) return;
    if (!scenarioId.trim() || !predicate.trim() || !knownEndpoint.trim()) {
      setError("Enter a scenario ID, predicate, and known endpoint.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.recover(activeDatasetId, {
        scenario_id: scenarioId.trim(),
        predicate: predicate.trim(),
        known_endpoint: knownEndpoint.trim(),
        missing_endpoint: missingEndpoint,
        limit: 20,
        strategy,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Recovery request failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader eyebrow="Causal inference / 04" title="Relationship recovery" description="Generate candidate links from a selected HAZOP scenario and inspect the evidence returned by the recovery API." actions={<Badge variant="outline">{activeDatasetId ?? "No dataset selected"}</Badge>} />
      <Card>
        <CardHeader><CardTitle>Recovery request</CardTitle><CardDescription>Recovered links are research proposals and are not automatically accepted into the knowledge graph.</CardDescription></CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-3 md:grid-cols-3">
            {strategies.map((item) => {
              const Icon = item.icon;
              const selected = strategy === item.id;
              return <Button key={item.id} type="button" variant={selected ? "secondary" : "outline"} aria-pressed={selected} onClick={() => setStrategy(item.id)} className={`h-auto min-h-24 justify-start whitespace-normal border p-3 text-left ${selected ? "border-primary" : "border-border"}`}><div className="flex items-start gap-3"><Icon className="mt-0.5 size-4 shrink-0" /><div><p className="text-xs font-semibold">{item.title}</p><p className="mt-1 text-[10px] font-normal leading-5 text-muted-foreground">{item.description}</p></div></div></Button>;
            })}
          </div>
          <Separator />
          <div className="grid gap-4 md:grid-cols-2">
            <label className="space-y-1.5"><span className="text-[11px] text-muted-foreground">Scenario ID</span><Input value={scenarioId} onChange={(e) => setScenarioId(e.target.value)} placeholder="e.g. S001" /></label>
            <label className="space-y-1.5"><span className="text-[11px] text-muted-foreground">Predicate / relationship type</span><Input value={predicate} onChange={(e) => setPredicate(e.target.value)} placeholder="e.g. leads_to_consequence" /></label>
            <label className="space-y-1.5"><span className="text-[11px] text-muted-foreground">Known endpoint</span><Input value={knownEndpoint} onChange={(e) => setKnownEndpoint(e.target.value)} placeholder="Known cause, deviation, or consequence" /></label>
            <label className="space-y-1.5"><span className="text-[11px] text-muted-foreground">Missing endpoint</span><select className="h-9 w-full border border-input bg-background px-3 text-xs" value={missingEndpoint} onChange={(e) => setMissingEndpoint(e.target.value as "subject" | "object")}><option value="object">Object — find the target</option><option value="subject">Subject — find the source</option></select></label>
          </div>
          {error && <p className="text-xs text-destructive" role="alert">{error}</p>}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4"><p className="text-[10px] leading-5 text-muted-foreground">Active dataset: {activeDatasetId ?? "Select a dataset in Datasets first."}</p><Button disabled={!activeDatasetId || loading} onClick={() => void runRecovery()}>{loading ? <LoaderCircle className="size-3.5 animate-spin" /> : <Search className="size-3.5" />}{loading ? "Running recovery…" : "Run recovery"}</Button></div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><div className="flex items-center justify-between gap-3"><div><CardTitle>Recovery response</CardTitle><CardDescription>Candidate relationships and reasoning from the backend.</CardDescription></div>{result && <Badge variant={result.validation.valid ? "secondary" : "destructive"}>{result.validation.valid ? "Validation passed" : "Validation failed"}</Badge>}</div></CardHeader>
        <CardContent>
          {!result ? <div className="flex min-h-48 flex-col items-center justify-center gap-3 border border-dashed border-border bg-muted/20 px-6 py-8 text-center"><Workflow className="size-6 text-muted-foreground" /><p className="text-xs font-medium">No recovery response yet</p><p className="max-w-sm text-[11px] leading-5 text-muted-foreground">Choose a dataset, enter the scenario details, and run a strategy to inspect returned candidates.</p></div> : <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">{[["Scenario", result.scenario_id], ["Strategy", result.strategy], ["Status", result.recovery.status]].map(([label, value]) => <div key={label} className="border border-border p-3"><p className="text-[10px] text-muted-foreground">{label}</p><p className="mt-1 break-words text-xs font-medium">{value}</p></div>)}</div>
            {result.recovery.candidates.length ? <div className="space-y-2"><h3 className="text-xs font-semibold">Retrieved candidates</h3>{result.recovery.candidates.map((candidate, index) => <pre key={index} className="overflow-auto whitespace-pre-wrap break-words border border-border bg-muted/20 p-3 text-[10px] leading-5">{JSON.stringify(candidate, null, 2)}</pre>)}</div> : <p className="text-xs text-muted-foreground">No retrieval candidates were returned for this request.</p>}
            {result.reasoning.proposed_relation && <div className="space-y-2"><h3 className="text-xs font-semibold">Proposed relationship</h3><pre className="overflow-auto whitespace-pre-wrap break-words border border-border p-3 text-[10px] leading-5">{JSON.stringify(result.reasoning.proposed_relation, null, 2)}</pre></div>}
            {(result.validation.errors.length > 0 || result.validation.warnings.length > 0) && <div className="space-y-1"><h3 className="text-xs font-semibold">Validation notes</h3>{[...result.validation.errors, ...result.validation.warnings].map((item, index) => <p key={index} className="text-[10px] text-muted-foreground">{item}</p>)}</div>}
            <div className="flex items-start gap-2 border border-border p-3"><ShieldAlert className="mt-0.5 size-4 shrink-0 text-muted-foreground" /><p className="text-[10px] leading-5 text-muted-foreground">{result.notice}</p></div>
          </div>}
        </CardContent>
      </Card>
    </div>
  );
}
