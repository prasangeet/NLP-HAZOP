import { useRef, useState, type ChangeEvent } from "react";
import {
  FileSearch,
  FileText,
  FileUp,
  X,
  ScanText,
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

const MAX_FILE_SIZE = 2 * 1024 * 1024;
const ALLOWED_EXTENSIONS = [".txt", ".md", ".log"];

export default function ReportExtraction() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (!file) return;

    const hasValidExtension = ALLOWED_EXTENSIONS.some((extension) =>
      file.name.toLowerCase().endsWith(extension),
    );

    if (!hasValidExtension) {
      setSelectedFile(null);
      setPreview(null);
      setError("Unsupported file type. Choose TXT, MD, or LOG.");
      return;
    }

    setSelectedFile(file);
    setPreview(null);
    setError(null);

    if (file.size > MAX_FILE_SIZE) {
      setError("The selected file exceeds the 2 MB limit.");
      return;
    }

    try {
      const contents = await file.text();
      setPreview(contents.slice(0, 2400));
    } catch {
      setError("The selected report could not be read.");
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreview(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const fileIsValid =
    selectedFile !== null &&
    selectedFile.size <= MAX_FILE_SIZE &&
    ALLOWED_EXTENSIONS.some((extension) =>
      selectedFile.name.toLowerCase().endsWith(extension),
    );

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Document ingestion / 02"
        title="Report extraction"
        description="Prepare unstructured process-safety reports for entity, relationship, and evidence extraction."
        actions={<Badge variant="outline">TXT · MD · LOG</Badge>}
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(300px,0.8fr)_minmax(0,1.2fr)]">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <FileUp className="size-4 text-muted-foreground" />
              <CardTitle>Source report</CardTitle>
            </div>
            <CardDescription>
              Choose a text-based report to inspect before extraction.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <Input
              ref={fileInputRef}
              type="file"
              accept=".txt,.md,.log,text/plain,text/markdown"
              className="sr-only"
              aria-label="Choose process safety report"
              onChange={handleFileChange}
            />

            <div className="flex flex-col items-center gap-3 border border-dashed border-border bg-muted/20 px-5 py-9 text-center">
              <div className="flex size-10 items-center justify-center border border-border bg-background">
                <FileText className="size-5 text-muted-foreground" />
              </div>

              <div className="space-y-1">
                <p className="text-xs font-medium">
                  {selectedFile ? "Report selected" : "Select a source report"}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  Maximum file size: 2 MB
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
                <FileText className="size-4 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-medium">
                    {selectedFile.name}
                  </p>
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                <Badge variant={fileIsValid ? "secondary" : "destructive"}>
                  {fileIsValid ? "Ready" : "Invalid"}
                </Badge>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Remove selected report"
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

            <div className="space-y-3 border-t border-border pt-4">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] text-muted-foreground">
                  Extraction endpoint
                </span>
                <Badge variant="outline">Not connected</Badge>
              </div>

              <Button className="w-full" disabled>
                <ScanText className="size-3.5" />
                Extraction integration pending
              </Button>

              <p className="text-[10px] leading-5 text-muted-foreground">
                The selected report can be inspected locally. This button
                remains unavailable until the extraction API is wired to the client.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <FileSearch className="size-4 text-muted-foreground" />
                <CardTitle>Document preview</CardTitle>
              </div>
              {preview !== null ? (
                <Badge variant="secondary">Local preview</Badge>
              ) : null}
            </div>
            <CardDescription>
              A read-only excerpt from the selected source file.
            </CardDescription>
          </CardHeader>

          <CardContent>
            {preview !== null ? (
              <div className="space-y-2">
                <pre className="min-h-80 max-h-[520px] overflow-auto whitespace-pre-wrap break-words border border-border bg-muted/30 p-4 text-[10px] leading-5 text-foreground">
                  {preview || "The selected report is empty."}
                </pre>
                {preview.length >= 2400 ? (
                  <p className="text-[10px] text-muted-foreground">
                    Preview truncated to the first 2,400 characters.
                  </p>
                ) : null}
              </div>
            ) : (
              <div className="flex min-h-80 flex-col items-center justify-center gap-3 border border-dashed border-border bg-muted/20 px-6 py-10 text-center">
                <FileSearch className="size-7 text-muted-foreground" />
                <div className="max-w-sm space-y-1">
                  <p className="text-xs font-medium">No report loaded</p>
                  <p className="text-[11px] leading-5 text-muted-foreground">
                    Select a TXT, MD, or LOG file to inspect its contents here.
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
