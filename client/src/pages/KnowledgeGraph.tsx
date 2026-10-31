import { useEffect, useState } from "react";
import { Background, Controls, MiniMap, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Database, GitBranch, LoaderCircle, RefreshCw, Workflow } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import PageHeader from "@/components/layout/PageHeader";
import { api, type GraphEdge, type GraphNode } from "@/lib/api";
import { useWorkspaceStore } from "@/lib/workspace-store";

export default function KnowledgeGraph() {
  const activeDatasetId = useWorkspaceStore((state) => state.activeDatasetId);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedItem, setSelectedItem] = useState<GraphNode | GraphEdge | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [counts, setCounts] = useState({ nodes: 0, edges: 0, truncated: false });

  useEffect(() => {
    if (!activeDatasetId) {
      setNodes([]);
      setEdges([]);
      setCounts({ nodes: 0, edges: 0, truncated: false });
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    api.graph(activeDatasetId).then((graph) => {
      if (cancelled) return;
      const graphNodes: Node[] = graph.nodes.map((node, index) => ({
        id: node.id,
        position: { x: (index % 5) * 220, y: Math.floor(index / 5) * 120 },
        data: { label: node.label },
        type: "default",
      }));
      const graphEdges: Edge[] = graph.edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.label,
        animated: false,
        data: { ...edge },
      }));
      setNodes(graphNodes);
      setEdges(graphEdges);
      setCounts({ nodes: graph.node_count, edges: graph.edge_count, truncated: graph.truncated });
    }).catch((err: unknown) => {
      if (!cancelled) setError(err instanceof Error ? err.message : "Could not load graph.");
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [activeDatasetId]);

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader eyebrow="Structural exploration / 03" title="Knowledge graph" description="Explore typed HAZOP entities and relationships returned by the graph API." actions={<Badge variant="outline">{activeDatasetId ?? "No dataset selected"}</Badge>} />
      {!activeDatasetId ? (
        <Card><CardContent className="flex min-h-72 flex-col items-center justify-center gap-3 text-center"><Database className="size-7 text-muted-foreground" /><p className="text-sm font-medium">Select a dataset first</p><p className="max-w-sm text-xs text-muted-foreground">Open Datasets and select an imported dataset. The graph is loaded from the configured backend.</p></CardContent></Card>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
          <Card className="min-w-0">
            <CardHeader><div className="flex items-center justify-between gap-3"><div><CardTitle>Graph explorer</CardTitle><CardDescription>{activeDatasetId}</CardDescription></div><Button variant="outline" size="sm" disabled={loading} onClick={() => { setError(null); setLoading(true); api.graph(activeDatasetId).then((graph) => { setNodes(graph.nodes.map((node, index) => ({ id: node.id, position: { x: (index % 5) * 220, y: Math.floor(index / 5) * 120 }, data: { label: node.label } }))); setEdges(graph.edges.map((edge) => ({ id: edge.id, source: edge.source, target: edge.target, label: edge.label, data: { ...edge } }))); setCounts({ nodes: graph.node_count, edges: graph.edge_count, truncated: graph.truncated }); }).catch((err: unknown) => setError(err instanceof Error ? err.message : "Could not load graph.")).finally(() => setLoading(false)); }}><RefreshCw className="size-3.5" />Refresh</Button></div></CardHeader>
            <CardContent>
              {error && <p className="mb-3 text-xs text-destructive" role="alert">{error}</p>}
              {loading ? <div className="flex min-h-[420px] items-center justify-center gap-2 text-xs text-muted-foreground"><LoaderCircle className="size-4 animate-spin" />Loading graph…</div> : nodes.length === 0 ? <div className="flex min-h-[420px] items-center justify-center border border-dashed border-border text-center"><div><GitBranch className="mx-auto size-7 text-muted-foreground" /><p className="mt-3 text-xs font-medium">No relationships found</p><p className="mt-1 text-[11px] text-muted-foreground">The selected dataset may contain no mapped causal relationships.</p></div></div> : <div className="h-[560px] border border-border"><ReactFlow nodes={nodes} edges={edges} fitView onNodeClick={(_, node) => setSelectedItem({ id: node.id, label: String(node.data.label), entity_type: "Entity", dataset_id: activeDatasetId })} onEdgeClick={(_, edge) => setSelectedItem((edge.data ?? null) as GraphEdge | null)}><Background /><Controls /><MiniMap pannable zoomable /></ReactFlow></div>}
              {counts.truncated && <p className="mt-2 text-[10px] text-muted-foreground">The backend limited this graph response. Increase the API limit to inspect more edges.</p>}
            </CardContent>
          </Card>
          <div className="space-y-4">
            <Card><CardHeader><div className="flex items-center gap-2"><Workflow className="size-4 text-muted-foreground" /><CardTitle>Graph contents</CardTitle></div><CardDescription>Counts from the selected dataset.</CardDescription></CardHeader><CardContent className="space-y-3">{[["Nodes", counts.nodes], ["Edges", counts.edges]].map(([label, value]) => <div key={label} className="flex items-center justify-between border-b border-border pb-2 last:border-0"><span className="text-xs text-muted-foreground">{label}</span><span className="text-sm font-semibold tabular-nums">{value}</span></div>)}</CardContent></Card>
            <Card><CardHeader><div className="flex items-center gap-2"><Database className="size-4 text-muted-foreground" /><CardTitle>Selection details</CardTitle></div></CardHeader><CardContent>{selectedItem ? <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-[10px] leading-5">{JSON.stringify(selectedItem, null, 2)}</pre> : <p className="text-[11px] leading-5 text-muted-foreground">Select a node or edge to inspect its attributes and evidence.</p>}</CardContent></Card>
          </div>
        </div>
      )}
    </div>
  );
}
