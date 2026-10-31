import { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  Database,
  FileSearch,
  FlaskConical,
  GitBranch,
  Workflow,
} from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import PageHeader from "@/components/layout/PageHeader";
import { api } from "@/lib/api";
import { useWorkspaceStore } from "@/lib/workspace-store";

const causalStages = [
  { number: "01", label: "Deviation" },
  { number: "02", label: "Cause" },
  { number: "03", label: "Consequence" },
  { number: "04", label: "Safeguard" },
];

const modules = [
  {
    number: "01",
    title: "Datasets",
    description: "Import and inspect structured HAZOP records.",
    icon: Database,
  },
  {
    number: "02",
    title: "Report extraction",
    description: "Turn unstructured process-safety reports into structured information.",
    icon: FileSearch,
  },
  {
    number: "03",
    title: "Knowledge graph",
    description: "Explore entities, causal links, provenance, and evidence.",
    icon: GitBranch,
  },
  {
    number: "04",
    title: "Relationship recovery",
    description: "Propose potentially missing links using different reasoning strategies.",
    icon: Workflow,
  },
  {
    number: "05",
    title: "Experiments",
    description: "Evaluate recovery methods against reference relationships.",
    icon: FlaskConical,
  },
];

export default function Overview() {
  const activeDatasetId = useWorkspaceStore((state) => state.activeDatasetId);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");

  useEffect(() => {
    let cancelled = false;
    api.health()
      .then(() => { if (!cancelled) setApiStatus("online"); })
      .catch(() => { if (!cancelled) setApiStatus("offline"); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Research workspace / 00"
        title="NLP for structured HAZOP reasoning"
        description="A workspace for representing, exploring, and recovering causal relationships in process-hazard analysis."
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(260px,0.8fr)]">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-1">
                <CardTitle>Target causal structure</CardTitle>
                <CardDescription>
                  The relationship chain this project aims to preserve and reason over.
                </CardDescription>
              </div>
              <Badge variant="outline">Core model</Badge>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="grid gap-2 sm:grid-cols-4">
              {causalStages.map((stage, index) => (
                <div key={stage.number} className="flex min-w-0 items-center gap-2">
                  <div className="min-w-0 flex-1 border border-border bg-muted/30 px-3 py-3">
                    <p className="text-[10px] tabular-nums text-muted-foreground">
                      {stage.number}
                    </p>
                    <p className="mt-2 truncate text-xs font-medium">
                      {stage.label}
                    </p>
                  </div>
                  {index < causalStages.length - 1 ? (
                    <ArrowRight className="hidden size-3 shrink-0 text-muted-foreground sm:block" />
                  ) : null}
                </div>
              ))}
            </div>

            <p className="text-[11px] leading-5 text-muted-foreground">
              Recovery focuses on relationships that may be missing from this chain;
              inferred links remain proposals until reviewed.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Activity className="size-4 text-muted-foreground" />
              <CardTitle>Workspace state</CardTitle>
            </div>
            <CardDescription>
              Current client-side context. No backend data is being presented as loaded.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="flex items-start justify-between gap-3 border-b border-border pb-3">
              <div>
                <p className="text-xs font-medium">Active dataset</p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  {activeDatasetId ?? "Choose one in Datasets"}
                </p>
              </div>
              <Badge variant={activeDatasetId ? "secondary" : "outline"}>{activeDatasetId ? "Selected" : "None"}</Badge>
            </div>

            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-medium">API connection</p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  {api.baseUrl}
                </p>
              </div>
              <Badge variant={apiStatus === "online" ? "secondary" : "outline"}>
                {apiStatus === "checking" ? "Checking…" : apiStatus === "online" ? "Connected" : "Offline"}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      <section className="space-y-3">
        <div className="flex items-end justify-between gap-3">
          <div className="space-y-1">
            <h3 className="text-sm font-semibold">Research modules</h3>
            <p className="text-[11px] text-muted-foreground">
              Move through the workflow using the workspace navigation.
            </p>
          </div>
          <span className="text-[10px] tracking-widest text-muted-foreground">
            05 MODULES
          </span>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {modules.map((module) => {
            const Icon = module.icon;

            return (
              <Card key={module.number} className="h-full">
                <CardHeader className="flex flex-row items-start gap-3 space-y-0">
                  <div className="flex size-9 shrink-0 items-center justify-center border border-border bg-muted/30">
                    <Icon className="size-4" />
                  </div>

                  <div className="min-w-0 flex-1 space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <CardTitle>{module.title}</CardTitle>
                      <span className="text-[10px] tabular-nums text-muted-foreground">
                        {module.number}
                      </span>
                    </div>
                    <CardDescription>{module.description}</CardDescription>
                  </div>
                </CardHeader>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
