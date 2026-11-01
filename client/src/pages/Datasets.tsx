import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import { Database, FileSpreadsheet, FileUp, Eye, LoaderCircle, RefreshCw, CheckCircle2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import PageHeader from "@/components/layout/PageHeader";
import { api, type CsvPreview, type Dataset } from "@/lib/api";
import { useWorkspaceStore } from "@/lib/workspace-store";

export default function Datasets() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [loading, setLoading] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [viewingDataset, setViewingDataset] = useState<Dataset | null>(null);
  const [viewingScenarios, setViewingScenarios] = useState<import("@/lib/api").Scenario[]>([]);
  const [viewLoading, setViewLoading] = useState(false);
  const activeDatasetId = useWorkspaceStore((state) => state.activeDatasetId);
  const setActiveDataset = useWorkspaceStore((state) => state.setActiveDataset);

  const viewDataset = async (dataset: Dataset) => {
    setViewingDataset(dataset); setViewingScenarios([]); setViewLoading(true);
    try { const result = await api.listScenarios(dataset.dataset_id); setViewingScenarios(result.scenarios); }
    catch (err) { setError(err instanceof Error ? err.message : "Could not load dataset records."); }
    finally { setViewLoading(false); }
  };

  const loadDatasets = useCallback(async () => {
    setLoading(true);
    try {
      const result = await api.listDatasets();
      setDatasets(result.datasets);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load datasets.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadDatasets(); }, [loadDatasets]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreview(null);
    setNotice(null);
    setError(file.name.toLowerCase().endsWith(".csv") ? null : "Choose a CSV file.");
    const stem = file.name.replace(/\.csv$/i, "").replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "").slice(0, 80);
    setDatasetId(stem);
    setDatasetName(file.name.replace(/\.csv$/i, ""));
  };

  const previewFile = async () => {
    if (!selectedFile) return;
    setPreviewing(true);
    setError(null);
    setNotice(null);
    try {
      const result = await api.previewDataset(selectedFile);
      setPreview(result);
      setDatasetId(result.dataset_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dataset preview failed.");
    } finally {
      setPreviewing(false);
    }
  };

  const importFile = async () => {
    if (!selectedFile || !preview) return;
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const result = await api.importDataset(selectedFile, datasetId || preview.dataset_id, datasetName || selectedFile.name);
      setActiveDataset(result.dataset);
      setNotice(`Imported “${result.dataset.name}” with ${result.dataset.record_count} scenarios and ${result.dataset.relationship_count} relationships.`);
      setPreview(null);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await loadDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dataset import failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6">
      <PageHeader eyebrow="Input data / 01" title="Datasets" description="Preview, validate, and import structured HAZOP records into the configured backend." actions={<Badge variant="outline">{api.baseUrl}</Badge>} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(280px,0.85fr)]">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2"><FileUp className="size-4 text-muted-foreground" /><CardTitle>Dataset source</CardTitle></div>
            <CardDescription>Preview validates the CSV without saving it. Import persists it to the backend.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input ref={fileInputRef} type="file" accept=".csv,text/csv" className="sr-only" aria-label="Choose HAZOP CSV file" onChange={handleFileChange} />
            <div className="flex flex-col items-center justify-center gap-3 border border-dashed border-border bg-muted/20 px-5 py-8 text-center">
              <FileSpreadsheet className="size-6 text-muted-foreground" />
              <p className="text-xs font-medium">{selectedFile?.name ?? "Choose a HAZOP CSV file"}</p>
              <p className="text-[11px] text-muted-foreground">CSV · Maximum upload size 10 MB · UTF-8</p>
              <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}><FileUp className="size-3.5" />Browse files</Button>
            </div>

            {selectedFile && (
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1.5"><span className="text-[11px] text-muted-foreground">Dataset ID</span><Input value={datasetId} onChange={(e) => setDatasetId(e.target.value)} /></label>
                <label className="space-y-1.5"><span className="text-[11px] text-muted-foreground">Display name</span><Input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} /></label>
              </div>
            )}

            {error && <p className="text-xs text-destructive" role="alert">{error}</p>}
            {notice && <p className="text-xs text-foreground" role="status">{notice}</p>}

            <div className="flex flex-wrap gap-2 border-t border-border pt-4">
              <Button size="sm" variant="outline" disabled={!selectedFile || !selectedFile.name.toLowerCase().endsWith(".csv") || previewing} onClick={() => void previewFile()}>
                {previewing ? <LoaderCircle className="size-3.5 animate-spin" /> : <Eye className="size-3.5" />}Preview and validate
              </Button>
              <Button size="sm" disabled={!selectedFile || !preview || loading || !datasetId.trim()} onClick={() => void importFile()}>
                {loading ? <LoaderCircle className="size-3.5 animate-spin" /> : <Database className="size-3.5" />}Import dataset
              </Button>
            </div>

            {preview && (
              <div className="space-y-3 border-t border-border pt-4">
                <div className="flex flex-wrap items-center justify-between gap-2"><p className="text-xs font-semibold">Backend preview</p><Badge variant="secondary">{preview.status}</Badge></div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {[
                    ["Rows read", preview.summary.rows_read],
                    ["Scenarios", preview.summary.records_loaded],
                    ["Relationships", preview.summary.relationships_generated],
                    ["Skipped rows", preview.summary.rows_skipped],
                  ].map(([label, value]) => <div key={label} className="border border-border p-3"><p className="text-[10px] text-muted-foreground">{label}</p><p className="mt-1 text-lg font-semibold tabular-nums">{value}</p></div>)}
                </div>
                {preview.warnings.length > 0 && <div className="space-y-1"><p className="text-[11px] font-medium">Validation notes</p>{preview.warnings.map((warning, index) => <p key={index} className="text-[10px] leading-5 text-muted-foreground">• {warning}</p>)}</div>}
                <div className="overflow-x-auto border border-border">
                  <table className="w-full text-left text-[11px]">
                    <thead className="bg-muted/50"><tr>{["Scenario", "Equipment", "Deviation", "Causes", "Consequences"].map((header) => <th key={header} className="whitespace-nowrap px-3 py-2 font-medium">{header}</th>)}</tr></thead>
                    <tbody>{preview.records.slice(0, 8).map((record, index) => <tr key={record.scenario_id ?? index} className="border-t border-border"><td className="px-3 py-2">{record.scenario_id}</td><td className="px-3 py-2">{record.equipment || "—"}</td><td className="px-3 py-2">{record.deviation || "—"}</td><td className="px-3 py-2">{Array.isArray(record.causes) ? record.causes.join(", ") : "—"}</td><td className="px-3 py-2">{Array.isArray(record.consequences) ? record.consequences.join(", ") : "—"}</td></tr>)}</tbody>
                  </table>
                </div>
                {preview.preview_truncated && <p className="text-[10px] text-muted-foreground">The server limited the returned sample to {preview.preview_limit} records.</p>}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-2"><div className="flex items-center gap-2"><Database className="size-4 text-muted-foreground" /><CardTitle>Dataset inventory</CardTitle></div><Button variant="ghost" size="icon-sm" aria-label="Refresh dataset list" onClick={() => void loadDatasets()} disabled={loading}><RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} /></Button></div>
            <CardDescription>Datasets returned by the backend.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {datasets.length === 0 && !loading && <div className="border border-dashed border-border px-4 py-8 text-center"><Database className="mx-auto size-6 text-muted-foreground" /><p className="mt-2 text-xs font-medium">No imported datasets</p><p className="mt-1 text-[10px] text-muted-foreground">Import a valid CSV to make it available to other modules.</p></div>}
            {datasets.map((dataset) => {
              const active = activeDatasetId === dataset.dataset_id;
              return <div key={dataset.dataset_id} className={`space-y-2 border p-3 ${active ? "border-primary bg-muted/30" : "border-border"}`}>
                <button type="button" onClick={() => setActiveDataset(dataset)} className="w-full text-left">
                  <div className="flex items-start justify-between gap-2"><div className="min-w-0"><p className="truncate text-xs font-medium">{dataset.name}</p><p className="mt-1 truncate text-[10px] text-muted-foreground">{dataset.dataset_id}</p></div>{active ? <Badge><CheckCircle2 className="mr-1 size-3" />Active</Badge> : <Badge variant="outline">Select</Badge>}</div>
                  <div className="mt-3 flex gap-4 text-[10px] text-muted-foreground"><span>{dataset.record_count} scenarios</span><span>{dataset.relationship_count} relationships</span></div>
                </button>
                <Button size="sm" variant="outline" className="w-full" onClick={() => void viewDataset(dataset)}><Eye className="size-3.5" />View records</Button>
              </div>;
            })}
            {loading && <div className="flex items-center justify-center gap-2 py-3 text-xs text-muted-foreground"><LoaderCircle className="size-4 animate-spin" />Loading datasets…</div>}
          </CardContent>
        </Card>
      </div>
      {viewingDataset && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setViewingDataset(null); }}>
        <section role="dialog" aria-modal="true" aria-labelledby="dataset-dialog-title" className="flex max-h-[85vh] w-full max-w-6xl flex-col border border-border bg-background shadow-xl">
          <header className="flex items-start justify-between gap-3 border-b border-border p-4"><div><h2 id="dataset-dialog-title" className="text-sm font-semibold">{viewingDataset.name}</h2><p className="mt-1 text-[11px] text-muted-foreground">{viewingDataset.dataset_id} · {viewingDataset.record_count} scenarios · {viewingDataset.relationship_count} relationships</p></div><Button size="sm" variant="outline" onClick={() => setViewingDataset(null)}>Close</Button></header>
          <div className="min-h-0 flex-1 overflow-auto p-4">
            {viewLoading ? <div className="flex items-center gap-2 py-10 text-xs text-muted-foreground"><LoaderCircle className="size-4 animate-spin"/>Loading records…</div> : viewingScenarios.length === 0 ? <p className="py-10 text-center text-xs text-muted-foreground">No scenario records returned.</p> : <div className="overflow-x-auto border border-border"><table className="w-full min-w-[900px] text-left text-[11px]"><thead className="sticky top-0 bg-muted"><tr>{["Scenario","Equipment","Deviation","Causes","Consequences","Safeguards","Recommendations"].map((h)=><th key={h} className="p-3 font-medium">{h}</th>)}</tr></thead><tbody>{viewingScenarios.map((record,index)=><tr key={record.scenario_id || index} className="border-t border-border align-top"><td className="p-3 font-medium">{record.scenario_id}</td><td className="p-3">{record.equipment || "—"}</td><td className="p-3">{record.deviation || "—"}</td><td className="p-3">{record.causes?.join(" · ") || "—"}</td><td className="p-3">{record.consequences?.join(" · ") || "—"}</td><td className="p-3">{record.safeguards?.join(" · ") || "—"}</td><td className="p-3">{record.recommendations?.join(" · ") || "—"}</td></tr>)}</tbody></table></div>}
            <p className="mt-3 text-[10px] text-muted-foreground">The API currently returns up to 500 scenarios per request.</p>
          </div>
        </section>
      </div>}
    </div>
  );
}
