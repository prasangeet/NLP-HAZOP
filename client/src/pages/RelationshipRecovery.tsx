import { useState } from "react";
import { GitBranch, BrainCircuit, Workflow, Search } from "lucide-react";

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

const strategies = [
  {
    id: "graph",
    title: "Graph-based",
    description: "Infer missing links from existing graph structure.",
    icon: GitBranch,
  },
  {
    id: "llm",
    title: "LLM-based",
    description: "Use language-model reasoning to propose relationships.",
    icon: BrainCircuit,
  },
  {
    id: "hybrid",
    title: "Hybrid",
    description: "Combine graph reasoning and language-model inference.",
    icon: Workflow,
  },
] as const;

type RecoveryStrategy = (typeof strategies)[number]["id"];

export default function RelationshipRecovery() {
  const [strategy, setStrategy] = useState<RecoveryStrategy>("hybrid");

  return (
    <div className="mx-auto w-full max-w-6xl space-y-8">
      <section className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight">
          Relationship Recovery
        </h2>

        <p className="text-sm text-muted-foreground">
          Identify potentially missing causal links in structured HAZOP data.
        </p>
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recovery strategy</CardTitle>
          <CardDescription>
            Choose the method to use when recovering missing relationships.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            {strategies.map((item) => {
              const Icon = item.icon;
              const isSelected = strategy === item.id;

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setStrategy(item.id)}
                  aria-pressed={isSelected}
                  className={`flex flex-col items-start gap-3 rounded-lg border p-4 text-left transition-colors ${
                    isSelected
                      ? "border-primary bg-accent"
                      : "border-border hover:bg-accent/50"
                  }`}
                >
                  <Icon className="size-5" />

                  <div className="space-y-1">
                    <p className="text-sm font-medium">{item.title}</p>
                    <p className="text-xs leading-5 text-muted-foreground">
                      {item.description}
                    </p>
                  </div>

                  {isSelected && (
                    <Badge variant="secondary">Selected</Badge>
                  )}
                </button>
              );
            })}
          </div>

          <Separator />

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-medium">Selected method</p>
              <p className="text-sm text-muted-foreground">
                {strategies.find((item) => item.id === strategy)?.title}
              </p>
            </div>

            <Button disabled>
              <Search className="size-4" />
              Recover relationships
            </Button>
          </div>

          <p className="text-xs text-muted-foreground">
            Recovery will be enabled after a dataset is selected and the
            backend API is connected.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Proposed relationships</CardTitle>
          <CardDescription>
            Review inferred causal links and their supporting evidence.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="flex min-h-64 flex-col items-center justify-center gap-3 rounded-lg border border-dashed p-8 text-center">
            <GitBranch className="size-8 text-muted-foreground" />

            <div className="max-w-md space-y-1">
              <p className="text-sm font-medium">No proposals available</p>
              <p className="text-sm text-muted-foreground">
                Run recovery on an imported dataset to inspect proposed
                relationships, evidence, and provenance.
              </p>
            </div>

            <Badge variant="outline">Awaiting recovery</Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
