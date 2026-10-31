import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Graph from "graphology";
import Sigma from "sigma";
import {
  Database,
  Expand,
  GitBranch,
  LoaderCircle,
  Minimize,
  RefreshCw,
  Search,
  Workflow,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import PageHeader from "@/components/layout/PageHeader";
import { api, type GraphEdge, type GraphNode } from "@/lib/api";
import { useWorkspaceStore } from "@/lib/workspace-store";

const MAX_VISIBLE_NODES = 600;
const MAX_VISIBLE_EDGES = 1500;

type GraphPayload = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
  truncated: boolean;
};

type SelectedItem =
  | { kind: "node"; value: GraphNode }
  | { kind: "edge"; value: GraphEdge }
  | null;

function cssColor(variable: string, fallback: string): string {
  const probe = document.createElement("span");
  probe.style.color = `var(${variable})`;
  document.body.appendChild(probe);
  const value = getComputedStyle(probe).color.trim();
  probe.remove();

  // CSS theme tokens use OKLCH. Convert OKLCH to sRGB because Sigma's
  // WebGL color parser expects hex/rgb colors, not OKLCH strings.
  const match = value.match(/^oklch\\(\\s*([\\d.]+)%?\\s+([\\d.]+)\\s+([\\d.]+)(?:deg)?(?:\\s*\\/\\s*([\\d.]+)%?)?\\s*\\)$/i);
  if (match) {
    let L = Number(match[1]);
    if (value.match(/^oklch\\(\\s*[\\d.]+%/i)) L /= 100;
    const C = Number(match[2]);
    const h = (Number(match[3]) * Math.PI) / 180;
    const a = C * Math.cos(h);
    const b = C * Math.sin(h);
    const l1 = L + 0.3963377774 * a + 0.2158037573 * b;
    const m1 = L - 0.1055613458 * a - 0.0638541728 * b;
    const s1 = L - 0.0894841775 * a - 1.291485548 * b;
    const linearToSrgb = (v: number) => {
      const v3 = v * v * v;
      return 255 * (v3 <= 0.0031308 ? 12.92 * v3 : 1.055 * Math.pow(v3, 1 / 2.4) - 0.055);
    };
    const rgb = [
      linearToSrgb(4.0767416621 * l1 ** 3 - 3.3077115913 * m1 ** 3 + 0.2309699292 * s1 ** 3),
      linearToSrgb(-1.2684380046 * l1 ** 3 + 2.6097574011 * m1 ** 3 - 0.3413193965 * s1 ** 3),
      linearToSrgb(-0.0041960863 * l1 ** 3 - 0.7034186147 * m1 ** 3 + 1.707614701 * s1 ** 3),
    ].map((channel) => Math.round(Math.max(0, Math.min(255, channel))));
    return `#${rgb.map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
  }

  const rgbMatch = value.match(/^rgba?\\(\\s*(\\d+)\\D+(\\d+)\\D+(\\d+)/i);
  if (rgbMatch) {
    return `#${rgbMatch.slice(1, 4).map((channel) => Number(channel).toString(16).padStart(2, "0")).join("")}`;
  }
  if (/^#[\\da-f]{3}(?:[\\da-f]{3})?$/i.test(value)) return value;
  return fallback;
}

function createGraph(payload: GraphPayload) {
  const graph = new Graph({ multi: true, type: "directed" });
  const visibleNodes = payload.nodes.slice(0, MAX_VISIBLE_NODES);
  const nodeIds = new Set(visibleNodes.map((node) => node.id));
  const rootStyles = getComputedStyle(document.documentElement);
  const primary = cssColor("--chart-1", "#d97706");
  const muted = cssColor("--muted-foreground", "#a1a1aa");
  const edgeColor = cssColor("--muted-foreground", "#a1a1aa");
  const palette = [
    primary,
    cssColor("--chart-2", primary),
    cssColor("--chart-3", primary),
    cssColor("--chart-4", primary),
    cssColor("--chart-5", primary),
  ];
  void rootStyles;

  const columns = Math.max(1, Math.ceil(Math.sqrt(visibleNodes.length)));
  visibleNodes.forEach((node, index) => {
    const typeIndex = Math.abs([...node.entity_type].reduce((sum, char) => sum + char.charCodeAt(0), 0)) % palette.length;
    graph.addNode(node.id, {
      label: node.label || node.id,
      x: (index % columns) * 2.4 + ((Math.floor(index / columns) % 2) * 0.6),
      y: -Math.floor(index / columns) * 2.2,
      size: 5,
      color: palette[typeIndex],
      entityType: node.entity_type,
      datasetId: node.dataset_id,
      original: node,
    });
  });

  const visibleEdges = payload.edges
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    .slice(0, MAX_VISIBLE_EDGES);

  visibleEdges.forEach((edge, index) => {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) return;
    const edgeKey = graph.hasEdge(edge.source, edge.target)
      ? `${edge.id}::${index}`
      : edge.id;
    graph.addEdgeWithKey(edgeKey, edge.source, edge.target, {
      label: edge.label || edge.predicate,
      color: edgeColor,
      size: 2,
      original: edge,
    });
  });

  return graph;
}

export default function KnowledgeGraph() {
  const activeDatasetId = useWorkspaceStore((state) => state.activeDatasetId);
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<Sigma | null>(null);
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [selectedItem, setSelectedItem] = useState<SelectedItem>(null);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadGraph = useCallback(async () => {
    if (!activeDatasetId) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.graph(activeDatasetId);
      setPayload(result);
      setSelectedItem(null);
      setSearch("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load graph.");
    } finally {
      setLoading(false);
    }
  }, [activeDatasetId]);

  useEffect(() => {
    if (!activeDatasetId) {
      setPayload(null);
      setSelectedItem(null);
      return;
    }
    void loadGraph();
  }, [activeDatasetId, loadGraph]);

  const graph = useMemo(() => {
    if (!payload) return null;
    try {
      return createGraph(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not prepare graph data.");
      return null;
    }
  }, [payload]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !graph) return;

    const renderer = new Sigma(graph, container, {
      renderLabels: true,
      renderEdgeLabels: false,
      defaultEdgeType: "arrow",
      labelDensity: 0.07,
      labelGridCellSize: 90,
      labelRenderedSizeThreshold: 9,
      zIndex: true,
      allowInvalidContainer: false,
      defaultNodeColor: cssColor("--chart-1", "#d97706"),
      defaultEdgeColor: cssColor("--muted-foreground", "#a1a1aa"),
      stagePadding: 32,
      minCameraRatio: 0.04,
      maxCameraRatio: 12,
    });

    renderer.on("clickNode", ({ node }) => {
      const original = graph.getNodeAttribute(node, "original") as GraphNode;
      setSelectedItem({ kind: "node", value: original });
    });
    renderer.on("clickEdge", ({ edge }) => {
      const original = graph.getEdgeAttribute(edge, "original") as GraphEdge;
      setSelectedItem({ kind: "edge", value: original });
    });
    renderer.on("clickStage", () => setSelectedItem(null));

    rendererRef.current = renderer;
    renderer.getCamera().animatedReset({ duration: 250 });

    return () => {
      renderer.kill();
      if (rendererRef.current === renderer) rendererRef.current = null;
    };
  }, [graph, fullscreen]);

  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer || !graph) return;

    const query = search.trim().toLowerCase();
    graph.forEachNode((node) => {
      const label = String(graph.getNodeAttribute(node, "label")).toLowerCase();
      const matches = !query || label.includes(query);
      graph.setNodeAttribute(node, "highlighted", Boolean(query && matches));
      graph.setNodeAttribute(node, "color", matches
        ? (graph.getNodeAttribute(node, "color") as string)
        : cssColor("--muted-foreground", "#a1a1aa"));
      graph.setNodeAttribute(node, "size", query && matches ? 8 : 5);
    });
    graph.forEachEdge((edge) => {
      const source = graph.source(edge);
      const target = graph.target(edge);
      const sourceMatch = Boolean(graph.getNodeAttribute(source, "highlighted"));
      const targetMatch = Boolean(graph.getNodeAttribute(target, "highlighted"));
      graph.setEdgeAttribute(edge, "color", query && (sourceMatch || targetMatch)
        ? cssColor("--primary", "rgb(180, 120, 80)")
        : cssColor("--border", "#71717a"));
      graph.setEdgeAttribute(edge, "size", query && (sourceMatch || targetMatch) ? 1.8 : 0.7);
    });
    renderer.refresh();
  }, [search, graph]);

  useEffect(() => {
    if (!fullscreen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [fullscreen]);

  const selectItemDetails = selectedItem
    ? selectedItem.value
    : null;

  const graphPanel = (
    <Card className={fullscreen ? "flex h-full min-h-0 flex-col rounded-none border-0" : "min-w-0"}>
      <CardHeader className="shrink-0 space-y-3 pb-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <CardTitle>Graph explorer</CardTitle>
            <CardDescription className="mt-1 truncate">{activeDatasetId}</CardDescription>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-2">
            <Badge variant="outline">{payload?.node_count ?? 0} nodes</Badge>
            <Badge variant="outline">{payload?.edge_count ?? 0} edges</Badge>
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
        <div className="relative max-w-sm">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Find an entity…"
            aria-label="Search graph entities"
            className="pl-8"
          />
        </div>
      </CardHeader>
      <CardContent className={fullscreen ? "flex min-h-0 flex-1 flex-col p-0" : ""}>
        {error && <p className="mb-3 text-xs text-destructive" role="alert">{error}</p>}
        {loading && !payload ? (
          <div className="flex min-h-80 items-center justify-center gap-2 text-xs text-muted-foreground">
            <LoaderCircle className="size-4 animate-spin" />Loading graph…
          </div>
        ) : !payload || payload.nodes.length === 0 ? (
          <div className="flex min-h-80 flex-col items-center justify-center gap-3 border border-dashed border-border text-center">
            <GitBranch className="size-7 text-muted-foreground" />
            <p className="text-xs font-medium">No relationships found</p>
            <p className="max-w-sm text-[11px] text-muted-foreground">The selected dataset may not contain mapped causal relationships.</p>
          </div>
        ) : (
          <div
            ref={containerRef}
            className={fullscreen ? "min-h-0 flex-1" : "h-[560px]"}
            style={{ background: "var(--background)", position: "relative" }}
            aria-label="Interactive causal graph. Select nodes or edges to inspect details."
          />
        )}
        {payload && (payload.truncated || payload.nodes.length > MAX_VISIBLE_NODES || payload.edges.length > MAX_VISIBLE_EDGES) && (
          <p className="mt-2 text-[10px] text-muted-foreground">
            Showing up to {MAX_VISIBLE_NODES} nodes and {MAX_VISIBLE_EDGES} edges to protect memory. Narrow the graph or inspect a focused causal chain for larger datasets.
          </p>
        )}
      </CardContent>
    </Card>
  );

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      {!fullscreen && (
        <PageHeader
          eyebrow="Structural exploration / 03"
          title="Knowledge graph"
          description="Explore HAZOP entities and causal relationships. Search to highlight entities; select a node or edge to inspect its evidence."
          actions={<Badge variant="outline">{activeDatasetId ?? "No dataset selected"}</Badge>}
        />
      )}
      {!activeDatasetId ? (
        <Card>
          <CardContent className="flex min-h-72 flex-col items-center justify-center gap-3 text-center">
            <Database className="size-7 text-muted-foreground" />
            <p className="text-sm font-medium">Select a dataset first</p>
            <p className="max-w-sm text-xs text-muted-foreground">Open Datasets and select an imported dataset. The graph loads from the configured backend.</p>
          </CardContent>
        </Card>
      ) : fullscreen ? (
        <div className="fixed inset-0 z-50 bg-background p-3 sm:p-5">
          <div className="h-full w-full overflow-hidden">{graphPanel}</div>
        </div>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
          {graphPanel}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2"><Workflow className="size-4 text-muted-foreground" /><CardTitle>Graph contents</CardTitle></div>
                <CardDescription>Returned by the selected dataset.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {[["Nodes", payload?.node_count ?? 0], ["Edges", payload?.edge_count ?? 0]].map(([label, value]) => (
                  <div key={label} className="flex items-center justify-between border-b border-border pb-2 last:border-0">
                    <span className="text-xs text-muted-foreground">{label}</span>
                    <span className="text-sm font-semibold tabular-nums">{value}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2"><Database className="size-4 text-muted-foreground" /><CardTitle>Selection details</CardTitle></div>
                <CardDescription>{selectedItem?.kind === "node" ? "Entity" : selectedItem?.kind === "edge" ? "Relationship" : "Nothing selected"}</CardDescription>
              </CardHeader>
              <CardContent>
                {selectItemDetails ? (
                  <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words text-[10px] leading-5">{JSON.stringify(selectItemDetails, null, 2)}</pre>
                ) : (
                  <p className="text-[11px] leading-5 text-muted-foreground">Select a node or relationship in the graph to inspect its properties and evidence.</p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
