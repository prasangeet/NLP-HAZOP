import { useRef, useState, type ChangeEvent } from "react";
import {
  Database,
  FileSpreadsheet,
  FileUp,
  Eye,
  X,
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

export default function Datasets() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sample, setSample] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setSelectedFile(null);
      setSample(null);
      setError("Choose a CSV file to inspect.");
      return;
    }

    setSelectedFile(file);
    setSample(null);
    setError(null);
  };

  const inspectFile = async () => {
    if (!selectedFile) return;

    try {
      const contents = await selectedFile.text();
      const lines = contents.split(/\r?\n/).slice(0, 8);
      setSample(lines.join("\n") || "The selected file is empty.");
      setError(null);
    } catch {
      setError("The selected file could not be read.");
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setSample(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Input data / 01"
        title="Datasets"
        description="Bring structured HAZOP records into the workspace and inspect their source data before import."
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)]">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <FileUp className="size-4 text-muted-foreground" />
              <CardTitle>Dataset source</CardTitle>
            </div>
            <CardDescription>
              Select a CSV file containing HAZOP records.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <Input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              aria-label="Choose HAZOP CSV file"
              onChange={handleFileChange}
            />

            <div className="flex flex-col items-center justify-center gap-3 border border-dashed border-border bg-muted/20 px-5 py-10 text-center">
              <div className="flex size-10 items-center justify-center border border-border bg-background">
                <FileSpreadsheet className="size-5 text-muted-foreground" />
              </div>

              <div className="space-y-1">
                <p className="text-xs font-medium">
                  {selectedFile ? "File selected" : "Choose a HAZOP CSV file"}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  CSV format · File is read locally for the sample preview
                </p>
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
              >
                <FileUp className="size-3.5" />
                Browse files
              </Button>
            </div>

            {selectedFile ? (
              <div className="flex min-w-0 items-center gap-3 border border-border p-3">
                <FileSpreadsheet className="size-4 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-medium">
                    {selectedFile.name}
                  </p>
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                <Badge variant="outline">Selected</Badge>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Remove selected file"
                  onClick={clearFile}
                >
                  <X className="size-3.5" />
                </Button>
              </div>
            ) : null}

            {error ? (
              <p className="text-[11px] text-destructive" role="alert">
                {error}
              </p>
            ) : null}

            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4">
              <p className="max-w-md text-[10px] leading-5 text-muted-foreground">
                Local inspection does not import or persist the dataset.
                Backend preview and import are not connected in this client yet.
              </p>

              <Button
                size="sm"
                disabled={!selectedFile}
                onClick={inspectFile}
              >
                <Eye className="size-3.5" />
                Read sample
              </Button>
            </div>

            {sample !== null ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-medium">First 8 lines</p>
                  <Badge variant="secondary">Local preview</Badge>
                </div>
                <pre className="max-h-64 overflow-auto border border-border bg-muted/30 p-3 text-[10px] leading-5 text-foreground">
                  {sample}
                </pre>
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Database className="size-4 text-muted-foreground" />
              <CardTitle>Dataset inventory</CardTitle>
            </div>
            <CardDescription>
              Imported datasets available for analysis.
            </CardDescription>
          </CardHeader>

          <CardContent>
            <div className="flex min-h-64 flex-col items-center justify-center gap-3 border border-dashed border-border bg-muted/20 px-5 py-8 text-center">
              <Database className="size-7 text-muted-foreground" />
              <div className="max-w-xs space-y-1">
                <p className="text-xs font-medium">Inventory not loaded</p>
                <p className="text-[11px] leading-5 text-muted-foreground">
                  The dataset listing API has not been connected to this
                  screen. No server-side dataset state is shown here.
                </p>
              </div>
              <Badge variant="outline">Awaiting API connection</Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
