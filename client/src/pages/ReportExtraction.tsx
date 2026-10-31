import { useRef, useState, type ChangeEvent } from "react";
import { FileSearch, FileText, FileUp, X, ScanText, LoaderCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import PageHeader from "@/components/layout/PageHeader";
import { api } from "@/lib/api";
import { useWorkspaceStore } from "@/lib/workspace-store";

const MAX_FILE_SIZE = 2 * 1024 * 1024;
const ALLOWED_EXTENSIONS = [".txt", ".md", ".log"];

export default function ReportExtraction() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeDatasetId = useWorkspaceStore((state) => state.activeDatasetId);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const validExtension = ALLOWED_EXTENSIONS.some((extension) => file.name.toLowerCase().endsWith(extension));
    if (!validExtension) { setSelectedFile(null); setPreview(null); setError("Choose a TXT, MD, or LOG report."); return; }
    if (file.size > MAX_FILE_SIZE) { setSelectedFile(file); setPreview(null); setError("The selected file exceeds the 2 MB limit."); return; }
    setSelectedFile(file);
    setPreview(null);
    setResponse(null);
    setError(null);
    const stem = file.name.replace(/\.(txt|md|log)$/i, "").replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "").slice(0, 80);
    setDatasetId(stem || activeDatasetId || "report-extraction");
    try { setPreview((await file.text()).slice(0, 2400)); }
    catch { setError("The selected report could not be read."); }
  };

  const clearFile = () => {
    setSelectedFile(null); setPreview(null); setResponse(null); setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const fileIsValid = selectedFile !== null && selectedFile.size <= MAX_FILE_SIZE && ALLOWED_EXTENSIONS.some((extension) => selectedFile.name.toLowerCase().endsWith(extension));

  const runExtraction = async () => {
    if (!selectedFile || !fileIsValid || !datasetId.trim()) return;
    setLoading(true); setError(null); setResponse(null);
    try {
      const result = await api.extractReport(selectedFile, datasetId.trim(), selectedFile.name);
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Report extraction failed.");
    } finally { setLoading(false); }
  };

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader eyebrow="Document ingestion / 02" title="Report extraction" description="Extract structured HAZOP entities and relationships from an unstructured text report using the backend extraction pipeline." actions={<Badge variant="outline">TXT · MD · LOG · 2 MB max</Badge>} />
      <div className="grid gap-4 xl:grid-cols-[minmax(300px,0.8fr)_minmax(0,1.2fr)]">
        <Card>
          <CardHeader><div className="flex items-center gap-2"><FileUp className="size-4 text-muted-foreground" /><CardTitle>Source report</CardTitle></div><CardDescription>Extraction persists the result through the API. Use a new dataset ID to avoid collisions with existing imports.</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <Input ref={fileInputRef} type="file" accept=".txt,.md,.log,text/plain,text/markdown" className="sr-only" aria-label="Choose process safety report" onChange={(event) => void handleFileChange(event)} />
            <div className="flex flex-col items-center gap-3 border border-dashed border-border bg-muted/20 px-5 py-8 text-center"><FileText className="size-6 text-muted-foreground" /><p className="text-xs font-medium">{selectedFile?.name ?? "Select a source report"}</p><p className="text-[11px] text-muted-foreground">UTF-8 text · Maximum 2 MB</p><Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}><FileUp className="size-3.5" />Browse files</Button></div>
            {selectedFile && <div className="flex items-center gap-3 border border-border p-3"><div className="min-w-0 flex-1"><p className="truncate text-xs font-medium">{selectedFile.name}</p><p className="mt-1 text-[10px] text-muted-foreground">{(selectedFile.size / 1024).toFixed(1)} KB</p></div><Badge variant={fileIsValid ? "secondary" : "destructive"}>{fileIsValid ? "Ready" : "Invalid"}</Badge><Button variant="ghost" size="icon-sm" aria-label="Remove report" onClick={clearFile}><X className="size-3.5" /></Button></div>}
            {selectedFile && <label className="block space-y-1.5"><span className="text-[11px] text-muted-foreground">New dataset ID</span><Input value={datasetId} onChange={(event) => setDatasetId(event.target.value)} placeholder="report-dataset-id" /></label>}
            {activeDatasetId && <p className="text-[10px] text-muted-foreground">Currently selected workspace dataset: {activeDatasetId}. Extraction creates/persists a dataset using the ID above; it does not append to this dataset.</p>}
            {error && <p className="text-xs text-destructive" role="alert">{error}</p>}
            <div className="space-y-3 border-t border-border pt-4"><div className="flex items-center justify-between gap-2"><span className="text-[10px] text-muted-foreground">API endpoint</span><Badge variant="outline">POST /api/v1/reports/extract</Badge></div><Button className="w-full" disabled={!fileIsValid || !datasetId.trim() || loading} onClick={() => void runExtraction()}>{loading ? <LoaderCircle className="size-3.5 animate-spin" /> : <ScanText className="size-3.5" />}{loading ? "Extracting report…" : "Extract and persist"}</Button></div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><div className="flex items-center gap-2"><FileSearch className="size-4 text-muted-foreground" /><CardTitle>{response ? "Extraction response" : "Document preview"}</CardTitle></div><CardDescription>{response ? "Structured result returned by the backend." : "A local, read-only excerpt from the selected report."}</CardDescription></CardHeader>
          <CardContent>
            {error && response === null ? null : null}
            {response ? <pre className="max-h-[560px] overflow-auto whitespace-pre-wrap break-words border border-border bg-muted/30 p-4 text-[10px] leading-5">{JSON.stringify(response, null, 2)}</pre> : preview !== null ? <div className="space-y-2"><pre className="min-h-80 max-h-[520px] overflow-auto whitespace-pre-wrap break-words border border-border bg-muted/30 p-4 text-[10px] leading-5">{preview || "The selected report is empty."}</pre>{preview.length >= 2400 && <p className="text-[10px] text-muted-foreground">Preview truncated to the first 2,400 characters.</p>}</div> : <div className="flex min-h-80 flex-col items-center justify-center gap-3 border border-dashed border-border bg-muted/20 px-6 py-10 text-center"><FileSearch className="size-7 text-muted-foreground" /><p className="text-xs font-medium">No report loaded</p><p className="text-[11px] text-muted-foreground">Select a supported text file to inspect it here.</p></div>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
