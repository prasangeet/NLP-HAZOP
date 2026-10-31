import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { GitBranch, Database, FileSearch, Workflow } from "lucide-react";

const modules = [
  {
    title: "Datasets",
    description: "Import and inspect structured HAZOP records.",
    icon: Database,
    status: "Available",
  },
  {
    title: "Report Extraction",
    description: "Extract structured information from process-safety reports.",
    icon: FileSearch,
    status: "Available",
  },
  {
    title: "Knowledge Graph",
    description: "Explore causal entities and relationships.",
    icon: GitBranch,
    status: "Available",
  },
  {
    title: "Relationship Recovery",
    description: "Investigate potentially missing causal relationships.",
    icon: Workflow,
    status: "Available",
  },
];

export default function Overview() {
  return (
    <div className="mx-auto w-full max-w-6xl space-y-8">
      <section className="space-y-2">
        <Badge variant="outline">Research workspace</Badge>

        <h2 className="text-3xl font-semibold tracking-tight">
          NLP-HAZOP
        </h2>

        <p className="max-w-2xl text-sm leading-6 text-muted-foreground">
          Recover missing causal relationships in HAZOP studies using
          natural language processing, knowledge graphs, and reasoning
          methods.
        </p>
      </section>

      <section className="space-y-4">
        <div>
          <h3 className="text-lg font-medium">Workspace modules</h3>
          <p className="text-sm text-muted-foreground">
            Select a module from the sidebar to get started.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          {modules.map((module) => {
            const Icon = module.icon;

            return (
              <Card key={module.title}>
                <CardHeader className="flex flex-row items-start gap-4 space-y-0">
                  <div className="rounded-md border p-2">
                    <Icon className="size-5" />
                  </div>

                  <div className="flex-1 space-y-1">
                    <CardTitle className="text-base">
                      {module.title}
                    </CardTitle>
                    <CardDescription>
                      {module.description}
                    </CardDescription>
                  </div>
                </CardHeader>

                <CardContent>
                  <Badge variant="secondary">{module.status}</Badge>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
