import { useState } from "react";
import {
  Database,
  GitBranch,
  Network,
  Search,
  Workflow,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import PageHeader from "@/components/layout/PageHeader";

const graphElements = [
  { label: "Nodes", description: "HAZOP entities" },
  { label: "Edges", description: "Causal relationships" },
  { label: "Provenance", description: "Source and evidence" },
];

export default function KnowledgeGraph() {
  const [searchQuery, setSearchQuery] = useState("");

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Structural exploration / 03"
        title="Knowledge graph"
        description="Explore the entities and causal relationships represented in an imported HAZOP dataset."
        actions={<Badge variant="outline">No graph loaded</Badge>}
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
        <Card className="min-w-0">
          <CardHeader className="gap-3">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-1">
                <CardTitle>Graph explorer</CardTitle>
                <CardDescription>
                  Graph canvas and relationship inspection.
                </CardDescription>
              </div>

              <div className="flex items-center gap-2">
                <div className="relative min-w-0 flex-1 sm:w-64">
                  <Search className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={searchQuery}
                    onChange={(event) => setSearchQuery(event.target.value)}
                    className="h-8 pl-8"
                    placeholder="Search graph..."
                    aria-label="Search graph"
                    disabled
                  />
                </div>
                <Button variant="outline" size="sm" disabled>
                  <GitBranch className="size-3.5" />
                  Fit view
                </Button>
              </div>
            </div>
          </CardHeader>

          <CardContent>
            <div
              className="relative flex min-h-[420px] items-center justify-center overflow-hidden border border-border bg-muted/20 p-6"
              style={{
                backgroundImage:
                  "radial-gradient(var(--border) 1px, transparent 1px)",
                backgroundSize: "20px 20px",
              }}
            >
              <div className="flex max-w-sm flex-col items-center gap-3 bg-background/95 px-6 py-7 text-center">
                <div className="flex size-11 items-center justify-center border border-border bg-muted/30">
                  <Network className="size-5 text-muted-foreground" />
                </div>

                <Badge variant="outline">Awaiting dataset</Badge>

                <div className="space-y-1">
                  <h3 className="text-xs font-semibold">No graph to display</h3>
                  <p className="text-[11px] leading-5 text-muted-foreground">
                    Import a HAZOP dataset and load its graph to inspect nodes,
                    edges, predicates, and supporting evidence.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Workflow className="size-4 text-muted-foreground" />
                <CardTitle>Graph contents</CardTitle>
              </div>
              <CardDescription>
                Information available for inspection once a graph is loaded.
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-3">
              {graphElements.map((element) => (
                <div
                  key={element.label}
                  className="flex items-start justify-between gap-3 border-b border-border pb-3 last:border-0 last:pb-0"
                >
                  <div>
                    <p className="text-xs font-medium">{element.label}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">
                      {element.description}
                    </p>
                  </div>
                  <span className="text-[10px] text-muted-foreground">—</span>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Database className="size-4 text-muted-foreground" />
                <CardTitle>Selection details</CardTitle>
              </div>
              <CardDescription>
                Select a graph node or edge to inspect its attributes.
              </CardDescription>
            </CardHeader>

            <CardContent>
              <div className="border border-dashed border-border bg-muted/20 px-4 py-6 text-center">
                <p className="text-xs font-medium">Nothing selected</p>
                <p className="mt-1 text-[10px] leading-5 text-muted-foreground">
                  Entity properties, relationship type, provenance, and evidence
                  will be shown here.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
