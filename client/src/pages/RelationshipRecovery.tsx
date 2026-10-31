import { useState } from "react";
import {
  BrainCircuit,
  GitBranch,
  Search,
  Workflow,
  ShieldAlert,
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
import { Separator } from "@/components/ui/separator";
import PageHeader from "@/components/layout/PageHeader";

const strategies = [
  {
    id: "graph",
    number: "01",
    title: "Graph-based",
    description: "Use existing graph structure and multi-hop paths to propose missing links.",
    icon: GitBranch,
  },
  {
    id: "llm",
    number: "02",
    title: "LLM-based",
    description: "Use language-model reasoning over available HAZOP context.",
    icon: BrainCircuit,
  },
  {
    id: "hybrid",
    number: "03",
    title: "Hybrid",
    description: "Combine graph-based evidence with language-model proposals.",
    icon: Workflow,
  },
] as const;

type RecoveryStrategy = (typeof strategies)[number]["id"];

export default function RelationshipRecovery() {
  const [strategy, setStrategy] = useState<RecoveryStrategy>("hybrid");

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Causal inference / 04"
        title="Relationship recovery"
        description="Generate candidate causal links for incomplete HAZOP records and review their supporting evidence."
        actions={<Badge variant="outline">Proposal workflow</Badge>}
      />

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div className="space-y-1">
              <CardTitle>Recovery method</CardTitle>
              <CardDescription>
                Select the strategy used to generate candidate relationships.
              </CardDescription>
            </div>
            <Badge variant="secondary">Default: Hybrid</Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            {strategies.map((item) => {
              const Icon = item.icon;
              const isSelected = strategy === item.id;

              return (
                <Button
                  key={item.id}
                  type="button"
                  variant={isSelected ? "secondary" : "outline"}
                  aria-pressed={isSelected}
                  onClick={() => setStrategy(item.id)}
                  className={`h-auto min-h-28 w-full justify-start whitespace-normal border p-4 text-left ${isSelected ? "border-primary" : "border-border"}`}
                >
                  <div className="flex w-full items-start gap-3">
                    <div className="flex size-8 shrink-0 items-center justify-center border border-border bg-background">
                      <Icon className="size-4" />
                    </div>

                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-semibold">{item.title}</span>
                        <span className="text-[10px] tabular-nums text-muted-foreground">
                          {item.number}
                        </span>
                      </div>
                      <span className="block whitespace-normal text-[10px] font-normal leading-5 text-muted-foreground">
                        {item.description}
                      </span>
                      {isSelected ? (
                        <span className="block pt-1 text-[10px] font-medium text-foreground">
                          Selected method
                        </span>
                      ) : null}
                    </div>
                  </div>
                </Button>
              );
            })}
          </div>

          <Separator />

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <p className="text-xs font-medium">
                Selected: {strategies.find((item) => item.id === strategy)?.title}
              </p>
              <p className="text-[10px] text-muted-foreground">
                A dataset must be loaded before recovery can run.
              </p>
            </div>

            <Button disabled>
              <Search className="size-3.5" />
              Recovery API not connected
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <CardTitle>Candidate relationships</CardTitle>
              <CardDescription>
                Review proposed links, their provenance, and supporting evidence.
              </CardDescription>
            </div>
            <Badge variant="outline">No proposals</Badge>
          </div>
        </CardHeader>

        <CardContent>
          <div className="flex min-h-64 flex-col items-center justify-center gap-3 border border-dashed border-border bg-muted/20 px-6 py-10 text-center">
            <Workflow className="size-7 text-muted-foreground" />

            <div className="max-w-md space-y-1">
              <p className="text-xs font-semibold">No recovery run available</p>
              <p className="text-[11px] leading-5 text-muted-foreground">
                Once recovery is connected, proposed deviation, cause,
                consequence, and safeguard relationships will be listed here.
              </p>
            </div>
          </div>

          <div className="mt-4 flex items-start gap-2 border border-border px-3 py-3">
            <ShieldAlert className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            <p className="text-[10px] leading-5 text-muted-foreground">
              Recovered relationships are hypotheses, not verified process-safety
              facts. They must retain provenance and be reviewed before acceptance.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
