import { useState } from "react";
import { Database, FileUp, FileSpreadsheet } from "lucide-react";

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
import { Separator } from "@/components/ui/separator";

export default function Datasets() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  return (
    <div className="mx-auto w-full max-w-6xl space-y-8">
      <section className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight">Datasets</h2>
        <p className="text-sm text-muted-foreground">
          Import and inspect structured HAZOP records.
        </p>
      </section>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="rounded-md border p-2">
              <FileUp className="size-5" />
            </div>

            <div className="space-y-1">
              <CardTitle className="text-base">Import dataset</CardTitle>
              <CardDescription>
                Select a CSV file containing HAZOP records.
              </CardDescription>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <Input
            type="file"
            accept=".csv,text/csv"
            onChange={(event) =>
              setSelectedFile(event.target.files?.[0] ?? null)
            }
          />

          {selectedFile && (
            <div className="flex items-center gap-3 rounded-lg border p-3">
              <FileSpreadsheet className="size-5 shrink-0 text-muted-foreground" />

              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-muted-foreground">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </p>
              </div>

              <Badge variant="outline">Selected</Badge>
            </div>
          )}

          <Separator />

          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-muted-foreground">
              CSV import and validation will be connected to the backend next.
            </p>

            <Button disabled={!selectedFile} onClick={() => {}}>
              <Database className="size-4" />
              Preview dataset
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Imported datasets</CardTitle>
          <CardDescription>
            Datasets available for analysis will appear here.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed px-6 py-12 text-center">
            <Database className="size-8 text-muted-foreground" />

            <div className="space-y-1">
              <p className="text-sm font-medium">No datasets loaded</p>
              <p className="max-w-sm text-sm text-muted-foreground">
                Import a CSV dataset to begin exploring HAZOP scenarios and
                their causal relationships.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
