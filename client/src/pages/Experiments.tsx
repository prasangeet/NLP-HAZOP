import {
  ChartNoAxesCombined,
  ClipboardCheck,
  FlaskConical,
  Target,
  Scale,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import PageHeader from "@/components/layout/PageHeader";

const metrics = [
  {
    title: "Precision",
    symbol: "P",
    description: "How many proposed relationships are relevant or correct.",
    formula: "TP / (TP + FP)",
    icon: Target,
  },
  {
    title: "Recall",
    symbol: "R",
    description: "How many reference relationships the method successfully recovers.",
    formula: "TP / (TP + FN)",
    icon: ClipboardCheck,
  },
  {
    title: "F1-score",
    symbol: "F1",
    description: "Harmonic mean of precision and recall.",
    formula: "2 × (P × R) / (P + R)",
    icon: Scale,
  },
];

const methods = [
  { name: "Graph-based", detail: "Structural and multi-hop reasoning" },
  { name: "LLM-based", detail: "Language-model inference" },
  { name: "Hybrid", detail: "Combined graph and language-model reasoning" },
];

export default function Experiments() {
  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Validation / 05"
        title="Experiments"
        description="Compare relationship-recovery strategies using reference data and explicit evaluation criteria."
        actions={<Badge variant="outline">No evaluation run</Badge>}
      />

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <FlaskConical className="size-4 text-muted-foreground" />
            <CardTitle>Methods under evaluation</CardTitle>
          </div>
          <CardDescription>
            These are the recovery strategies available in the research workflow.
            No experiment results are loaded yet.
          </CardDescription>
        </CardHeader>

        <CardContent className="grid gap-3 md:grid-cols-3">
          {methods.map((method, index) => (
            <div key={method.name} className="border border-border p-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-semibold">{method.name}</p>
                <span className="text-[10px] tabular-nums text-muted-foreground">
                  0{index + 1}
                </span>
              </div>
              <p className="mt-2 text-[10px] leading-5 text-muted-foreground">
                {method.detail}
              </p>
              <div className="mt-4">
                <Badge variant="outline">Not evaluated</Badge>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <section className="space-y-3">
        <div className="space-y-1">
          <h3 className="text-sm font-semibold">Evaluation criteria</h3>
          <p className="text-[11px] text-muted-foreground">
            Metric definitions only; values will come from actual evaluation runs.
          </p>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          {metrics.map((metric) => {
            const Icon = metric.icon;

            return (
              <Card key={metric.title}>
                <CardHeader>
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <Icon className="size-4 text-muted-foreground" />
                      <CardTitle>{metric.title}</CardTitle>
                    </div>
                    <span className="font-heading text-sm text-muted-foreground">
                      {metric.symbol}
                    </span>
                  </div>
                  <CardDescription>{metric.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="border border-border bg-muted/20 px-3 py-2">
                    <p className="font-heading text-[11px]">{metric.formula}</p>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ChartNoAxesCombined className="size-4 text-muted-foreground" />
            <CardTitle>Experiment results</CardTitle>
          </div>
          <CardDescription>
            Precision, recall, F1-score, and method comparisons will appear after
            an evaluation run has been implemented.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="flex min-h-52 flex-col items-center justify-center gap-3 border border-dashed border-border bg-muted/20 px-6 py-8 text-center">
            <ChartNoAxesCombined className="size-7 text-muted-foreground" />
            <div className="space-y-1">
              <p className="text-xs font-semibold">No experiment results</p>
              <p className="max-w-sm text-[11px] leading-5 text-muted-foreground">
                Results will not be generated or estimated by the interface.
                They will be displayed when the evaluation pipeline supplies them.
              </p>
            </div>
            <Badge variant="outline">Awaiting evaluation pipeline</Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
