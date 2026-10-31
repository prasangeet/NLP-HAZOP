import {
  FlaskConical,
  ChartNoAxesCombined,
  ClipboardCheck,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function Experiments() {
  return (
    <div className="mx-auto w-full max-w-6xl space-y-8">
      <section className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight">
          Experiments
        </h2>

        <p className="text-sm text-muted-foreground">
          Evaluate and compare methods for recovering missing causal
          relationships in HAZOP studies.
        </p>
      </section>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="rounded-md border p-2">
              <FlaskConical className="size-5" />
            </div>

            <div className="space-y-1">
              <CardTitle className="text-base">
                Evaluation workspace
              </CardTitle>

              <CardDescription>
                Configure experiments and examine their evaluation results.
              </CardDescription>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          <div className="flex min-h-64 flex-col items-center justify-center gap-4 rounded-lg border border-dashed p-8 text-center">
            <ChartNoAxesCombined className="size-8 text-muted-foreground" />

            <div className="max-w-md space-y-2">
              <h3 className="text-base font-medium">
                No experiments available
              </h3>

              <p className="text-sm leading-6 text-muted-foreground">
                Evaluation results will appear here once the benchmarking
                workflow and its backend endpoints are available.
              </p>
            </div>

            <Badge variant="outline">Awaiting evaluation pipeline</Badge>
          </div>
        </CardContent>
      </Card>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <ClipboardCheck className="size-5 text-muted-foreground" />
              <CardTitle className="text-base">
                Evaluation metrics
              </CardTitle>
            </div>

            <CardDescription>
              Metrics for assessing recovered relationships.
            </CardDescription>
          </CardHeader>

          <CardContent>
            <p className="text-sm text-muted-foreground">
              Precision, recall, F1-score, and other relevant metrics can be
              displayed when evaluation data becomes available.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <FlaskConical className="size-5 text-muted-foreground" />
              <CardTitle className="text-base">
                Method comparison
              </CardTitle>
            </div>

            <CardDescription>
              Compare graph-based, LLM-based, and hybrid approaches.
            </CardDescription>
          </CardHeader>

          <CardContent>
            <p className="text-sm text-muted-foreground">
              Comparisons will be based on actual experiment outputs rather
              than predefined or fabricated results.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
