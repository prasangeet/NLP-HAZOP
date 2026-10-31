import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  type Edge,
  type Node,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Database,
  Expand,
  GitBranch,
  LoaderCircle,
  Minimize,
  RefreshCw,
  Workflow,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import PageHeader from "@/components/layout/PageHeader";
import { api, type GraphEdge, type GraphNode } from "@/lib/api";
import { useWorkspaceStore } from "@/lib/workspace-store";

type GraphPayload = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
  truncated: boolean;
};

function arrangeNodes(sourceNodes: GraphNode[]): Node[] {
  // Use a spacious staggered layout rather than packing every entity into a tight grid.
  const columns = Math.max(3, Math.ceil(Math.sqrt(sourceNodes.length * 0.8)));
  return sourceNodes.map((node, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    return {
      id: node.id,
      position: {
        x: column * 270 + (row % 2) * 90,
        y: row * 155,
      },
      data: { label: node.label },
      type: "default",
      style: {
        width: 190,
        minHeight: 56,
        padding: "12px 14px",
        borderRadius: "8px",
        border: "1px solid var(--border)",
        background: "var(--card)",
        color: "var(--card-foreground)",
        fontSize: "12px",
        lineHeight: 1.5,
        boxShadow: "0 2px 8px oklch(0 0 0 / 8%)",
      },
    };
  });
}

function buildEdges(sourceEdges: GraphEdge[], selectedEdgeId: string | null): Edge[] {
  return sourceEdges.map((edge) => {
    const selected = edge.id === selectedEdgeId;
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
      label: selected ? edge.label : undefined,
      labelStyle: { fill: "var(--foreground)", fontSize: 10 },
      labelBgStyle: { fill: "var(--background)", fillOpacity: 0.96 },
      style: {
        stroke: selected ? "var(--primary)" : "var(--muted-foreground)",
        strokeWidth: selected ? 2 : 1,
        opacity: selected ? 1 : 0.28,
      },
      markerEnd: { type: "arrowclosed" as const, color: selected ? "var(--primary)" : "var(--muted-foreground)" },
      data: { ...edge },
    };
  });
}

function GraphCanvasInner({
  nodes,
  edges,
  selectedEdgeId,
  onNodeSelect,
  onEdgeSelect,
  fullscreen,
}: {
  nodes: Node[];
  edges: Edge[];
  selectedEdgeId: string | null;
  onNodeSelect: (node: Node) => void;
  onEdgeSelect: (edge: Edge) => void;
  fullscreen: boolean;
}) {
  const { fitView } = useReactFlow();

  useEffect(() => {
    if (nodes.length) {
      const timer = window.setTimeout(() => void fitView({ padding: 0.18, duration: 250 }), 80);
      return () => window.clearTimeout(timer);
    }
  }, [nodes, fullscreen, fitView]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges.map((edge) => ({
        ...edge,
        label: edge.id === selectedEdgeId ? (edge.data as GraphEdge | undefined)?.label : undefined,
      }))}
      fitView
      minZoom={0.08}
      maxZoom={1.8}
      onNodeClick={(_, node) => onNodeSelect(node)}
      onEdgeClick={(_, edge) => onEdgeSelect(edge)}
      onPaneClick={() => onEdgeSelect({ id: "", data: undefined } as Edge)}
      proOptions={{ hideAttribution: true }}
      defaultEdgeOptions={{ type: "smoothstep" }}
    >
      <Background color="var(--border)" gap={22} size={1} />
      <Controls showInteractive={false} />
      <MiniMap
        pannable
        zoomable
        nodeColor={() => "var(--muted-foreground)"}
        maskColor="var(--background)"
        className="!border !border-border !bg-card"
      />
    </ReactFlow>
  );
}

function GraphCanvas(props: {
  nodes: Node[];
  edges: Edge[];
  selectedEdgeId: string | null;
  onNodeSelect: (node: Node) => void;
  onEdgeSelect: (edge: Edge) => void;
  fullscreen: boolean;
}) {
  return (
    <ReactFlowProvider>
      <GraphCanvasInner {...props} />
    </ReactFlowProvider>
  );
}

export default function KnowledgeGraph() {
  const activeDatasetId = useWorkspaceStore((state) => state.activeDatasetId);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [rawEdges, setRawEdges] = useState<GraphEdge[]>([]);
  const [selectedItem, setSelectedItem] = useState<GraphNode | GraphEdge | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [counts, setCounts] = useState({ nodes: 0, edges: 0, truncated: false });

  const edges = useMemo(() => buildEdges(rawEdges, selectedEdgeId), [rawEdges, selectedEdgeId]);

  const loadGraph = useCallback(async () => {
    if (!activeDatasetId) return;
    setLoading(true);
    setError(null);
    try {
      const graph: GraphPayload = await api.graph(activeDatasetId);
      setNodes(arrangeNodes(graph.nodes));
      setRawEdges(graph.edges);
      setSelectedItem(null);
      setSelectedEdgeId(null);
      setCounts({ nodes: graph.node_count, edges: graph.edge_count, truncated: graph.truncated });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load graph.");
    } finally {
      setLoading(false);
    }
  }, [activeDatasetId]);

  useEffect(() => {
    if (!activeDatasetId) {
      setNodes([]);
      setRawEdges([]);
      setCounts({ nodes: 0, edges: 0, truncated: false });
      setSelectedItem(null);
      return;
    }
    void loadGraph();
  }, [activeDatasetId, loadGraph]);

  useEffect(() => {
    if (!fullscreen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [fullscreen]);

  const selectNode = (node: Node) => {
    setSelectedEdgeId(null);
    setSelectedItem({
      id: node.id,
      label: String(node.data.label),
      entity_type: "Entity",
      dataset_id: activeDatasetId ?? "",
    } as GraphNode);
  };

  const selectEdge = (edge: Edge) => {
    if (!edge.id) {
      setSelectedEdgeId(null);
      setSelectedItem(null);
      return;
    }
    setSelectedEdgeId(edge.id);
    setSelectedItem((edge.data ?? null) as GraphEdge | null);
  };

  const graphPanel = (
    <Card className={fullscreen ? "flex h-full min-h-0 flex-col rounded-none border-0" : "min-w-0"}>
      <CardHeader className="shrink-0 space-y-0 pb-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <CardTitle>Graph explorer</CardTitle>
            <CardDescription className="mt-1 truncate">{activeDatasetId}</CardDescription>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Badge variant="outline">{counts.nodes} nodes</Badge>
            <Badge variant="outline">{counts.edges} edges</Badge>
            <Button variant="outline" size="sm" disabled={loading} onClick={() => void loadGraph()}>
              {loading ? <LoaderCircle className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
              Refresh
            </Button>
            <Button variant="outline" size="sm" onClick={() => setFullscreen((value) => !value)}>
              {fullscreen ? <Minimize className="size-3.5" /> : <Expand className="size-3.5" />}
              {fullscreen ? "Exit full screen" : "Full screen"}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className={fullscreen ? "flex min-h-0 flex-1 flex-col p-0" : ""}>
        {error && <p className="mb-3 text-xs text-destructive" role="alert">{error}</p>}
        {loading ? (
          <div className={fullscreen ? "flex min-h-0 flex-1 items-center justify-center gap-2 text-xs text-muted-foreground" : "flex min-h-[420px] items-center justify-center gap-2 text-xs text-muted-foreground"}>
            <LoaderCircle className="size-4 animate-spin" />Loading graph…
          </div>
        ) : nodes.length === 0 ? (
          <div className="flex min-h-80 flex-col items-center justify-center gap-3 border border-dashed border-border text-center">
            <GitBranch className="size-7 text-muted-foreground" />
            <p className="text-xs font-medium">No relationships found</p>
            <p className="max-w-sm text-[11px] text-muted-foreground">The selected dataset may contain no mapped causal relationships.</p>
          </div>
        ) : (
          <div className={fullscreen ? "min-h-0 flex-1" : "h-[560px]"} style={{ background: "var(--background)" }}>
            <GraphCanvas
              nodes={nodes}
              edges={edges}
              selectedEdgeId={selectedEdgeId}
              onNodeSelect={selectNode}
              onEdgeSelect={selectEdge}
              fullscreen={fullscreen}
            />
          </div>
        )}
        {counts.truncated && !fullscreen && <p className="mt-2 text-[10px] text-muted-foreground">The API limited this graph response. Use the graph filters or increase the backend limit to inspect more entities.</p>}
        {!fullscreen && nodes.length > 0 && <p className="mt-2 text-[10px] text-muted-foreground">Relationship labels are hidden to reduce clutter. Select an edge to highlight it and show its predicate. Use scroll to zoom and drag to pan.</p>}
      </CardContent>
    </Card>
  );

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      {!fullscreen && <PageHeader eyebrow="Structural exploration / 03" title="Knowledge graph" description="Explore typed HAZOP entities and relationships returned by the graph API." actions={<Badge variant="outline">{activeDatasetId ?? "No dataset selected"}</Badge>} />}
      {!activeDatasetId ? (
        <Card><CardContent className="flex min-h-72 flex-col items-center justify-center gap-3 text-center"><Database className="size-7 text-muted-foreground" /><p className="text-sm font-medium">Select a dataset first</p><p className="max-w-sm text-xs text-muted-foreground">Open Datasets and select an imported dataset. The graph is loaded from the configured backend.</p></CardContent></Card>
      ) : fullscreen ? (
        <div className="fixed inset-0 z-50 bg-background p-3 sm:p-5">
          <div className="h-full w-full overflow-hidden">{graphPanel}</div>
        </div>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
          {graphPanel}
          <div className="space-y-4">
            <Card>
              <CardHeader><div className="flex items-center gap-2"><Workflow className="size-4 text-muted-foreground" /><CardTitle>Graph contents</CardTitle></div><CardDescription>Counts from the selected dataset.</CardDescription></CardHeader>
              <CardContent className="space-y-3">{[["Nodes", counts.nodes], ["Edges", counts.edges]].map(([label, value]) => <div key={label} className="flex items-center justify-between border-b border-border pb-2 last:border-0"><span className="text-xs text-muted-foreground">{label}</span><span className="text-sm font-semibold tabular-nums">{value}</span></div>)}</CardContent>
            </Card>
            <Card>
              <CardHeader><div className="flex items-center gap-2"><Database className="size-4 text-muted-foreground" /><CardTitle>Selection details</CardTitle></div></CardHeader>
              <CardContent>{selectedItem ? <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-[10px] leading-5">{JSON.stringify(selectedItem, null, 2)}</pre> : <p className="text-[11px] leading-5 text-muted-foreground">Select a node or edge to inspect its attributes and evidence.</p>}</CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
