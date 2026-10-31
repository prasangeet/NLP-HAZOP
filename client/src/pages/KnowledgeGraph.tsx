import { GitBranch, Network, Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

export default function KnowledgeGraph() {
  return (
    <div className="mx-auto w-full max-w-6xl space-y-8">
      <section className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight">
          Knowledge Graph
        </h2>

        <p className="text-sm text-muted-foreground">
          Explore HAZOP entities and the causal relationships connecting them.
        </p>
      </section>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="rounded-md border p-2">
              <Network className="size-5" />
            </div>

            <div className="space-y-1">
              <CardTitle className="text-base">Graph explorer</CardTitle>
              <CardDescription>
                Inspect nodes, edges, and their supporting evidence.
              </CardDescription>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 sm:flex-row">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />

              <Input
                className="pl-9"
                placeholder="Search entities or relationships..."
                aria-label="Search graph"
              />
            </div>

            <Button variant="outline" disabled>
              <GitBranch className="size-4" />
              Explore graph
            </Button>
          </div>

          <div className="flex min-h-80 flex-col items-center justify-center gap-4 rounded-lg border border-dashed p-8 text-center">
            <div className="rounded-full border p-4">
              <Network className="size-8 text-muted-foreground" />
            </div>

            <div className="max-w-md space-y-2">
              <h3 className="text-base font-medium">No graph loaded</h3>

              <p className="text-sm leading-6 text-muted-foreground">
                Import a HAZOP dataset to explore its entities and causal
                relationships. Graph visualization and search will be enabled
                when the backend integration is connected.
              </p>
            </div>

            <Badge variant="outline">Awaiting dataset</Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
