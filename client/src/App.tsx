import { useState } from "react";

import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";

import AppHeader from "@/components/layout/AppHeader";
import AppSidebar, { type PageId } from "@/components/layout/AppSidebar";

import Overview from "@/pages/Overview";
import Datasets from "@/pages/Datasets";
import ReportExtraction from "@/pages/ReportExtraction";
import KnowledgeGraph from "@/pages/KnowledgeGraph";
import RelationshipRecovery from "@/pages/RelationshipRecovery";
import Experiments from "@/pages/Experiments";

const pageTitles: Record<PageId, string> = {
  overview: "Overview",
  datasets: "Datasets",
  "report-extraction": "Report Extraction",
  "knowledge-graph": "Knowledge Graph",
  "relationship-recovery": "Relationship Recovery",
  experiments: "Experiments",
};

function Workspace() {
  const [activePage, setActivePage] = useState<PageId>("overview");

  const renderPage = () => {
    switch (activePage) {
      case "overview":
        return <Overview />;
      case "datasets":
        return <Datasets />;
      case "report-extraction":
        return <ReportExtraction />;
      case "knowledge-graph":
        return <KnowledgeGraph />;
      case "relationship-recovery":
        return <RelationshipRecovery />;
      case "experiments":
        return <Experiments />;
    }
  };

  return (
    <TooltipProvider>
      <SidebarProvider>
        <AppSidebar activePage={activePage} onNavigate={setActivePage} />

        <SidebarInset className="min-h-svh">
          <AppHeader title={pageTitles[activePage]} />

          <main className="min-w-0 flex-1 overflow-auto p-4 md:p-6 lg:p-8">
            {renderPage()}
          </main>
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  );
}

export default function App() {
  return (
    <ThemeProvider defaultTheme="system">
      <Workspace />
    </ThemeProvider>
  );
}
