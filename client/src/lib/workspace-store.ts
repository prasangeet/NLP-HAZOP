import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { Dataset } from "@/lib/api";

interface WorkspaceState {
  activeDatasetId: string | null;
  setActiveDataset: (dataset: Dataset | null) => void;
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      activeDatasetId: null,
      setActiveDataset: (dataset) => set({ activeDatasetId: dataset?.dataset_id ?? null }),
    }),
    { name: "nlp-hazop-workspace" },
  ),
);
