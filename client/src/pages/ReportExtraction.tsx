import { useState } from "react";
import { FileSearch, FileText, Upload, X } from "lucide-react";

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

export default function ReportExtraction() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (file?: File) => {
    if (!file) return;

    const allowedExtensions = [".txt", ".md", ".log"];
    const hasValidExtension = allowedExtensions.some((extension) =>
      file.name.toLowerCase().endsWith(extension),
    );

    if (!hasValidExtension) {
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  };

  return (
    <div className="mx-auto w-full max-w-6xl space-y-8">
      <section className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight">
          Report Extraction
        </h2>

        <p className="text-sm text-muted-foreground">
          Extract structured information and causal relationships from
          unstructured process-safety reports.
        </p>
      </section>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="rounded-md border p-2">
              <FileSearch className="size-5" />
            </div>

            <div className="space-y-1">
              <CardTitle className="text-base">Source report</CardTitle>
              <CardDescription>
                Choose a text-based process-safety report for extraction.
              </CardDescription>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <Input
            type="file"
            accept=".txt,.md,.log,text/plain,text/markdown"
            onChange={(event) =>
              handleFileChange(event.target.files?.[0])
            }
          />

          {selectedFile && (
            <div className="flex items-center gap-3 rounded-lg border p-3">
              <FileText className="size-5 shrink-0 text-muted-foreground" />

              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">
                  {selectedFile.name}
                </p>

                <p className="text-xs text-muted-foreground">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </p>
              </div>

              <Badge variant="outline">Ready</Badge>

              <Button
                variant="ghost"
                size="icon"
                aria-label="Remove selected report"
                onClick={() => setSelectedFile(null)}
              >
                <X className="size-4" />
              </Button>
            </div>
          )}

          <Separator />

          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-muted-foreground">
              Supported formats: TXT, MD, LOG · Maximum size: 2 MB
            </p>

            <Button
              disabled={!selectedFile || selectedFile.size > 2 * 1024 * 1024}
              onClick={() => {}}
            >
              <Upload className="size-4" />
              Extract information
            </Button>
          </div>

          {selectedFile && selectedFile.size > 2 * 1024 * 1024 && (
            <p className="text-sm text-destructive">
              The selected file exceeds the 2 MB limit.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Extraction results</CardTitle>
          <CardDescription>
            Extracted entities, causal relationships, and supporting evidence
            will appear here.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed px-6 py-12 text-center">
            <FileSearch className="size-8 text-muted-foreground" />

            <div className="space-y-1">
              <p className="text-sm font-medium">No extraction results</p>

              <p className="max-w-md text-sm text-muted-foreground">
                Select a report to get started. Extraction results will be
                displayed here once the backend integration is implemented.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
