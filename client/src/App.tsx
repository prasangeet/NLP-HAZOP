import { useState } from "react";
import { Moon, Sun, Monitor } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider, useTheme } from "@/components/theme-provider";

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

function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Change appearance"
          title="Change appearance"
        >
          {theme === "system" ? (
            <Monitor className="size-4" />
          ) : theme === "dark" ? (
            <Moon className="size-4" />
          ) : (
            <Sun className="size-4" />
          )}
        </Button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}>
          <Sun className="size-4" />
          Light
          {theme === "light" && <span className="ml-auto">✓</span>}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}>
          <Moon className="size-4" />
          Dark
          {theme === "dark" && <span className="ml-auto">✓</span>}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}>
          <Monitor className="size-4" />
          System
          {theme === "system" && <span className="ml-auto">✓</span>}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

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
          <header className="sticky top-0 z-10 flex h-14 shrink-0 items-center gap-3 border-b bg-background/95 px-4 backdrop-blur">
            <SidebarTrigger />
            <Separator orientation="vertical" className="h-4" />

            <div className="min-w-0">
              <p className="truncate text-xs font-medium">
                {pageTitles[activePage]}
              </p>
            </div>

            <div className="ml-auto flex items-center gap-2">
              <span className="hidden text-[10px] text-muted-foreground sm:inline">
                RESEARCH WORKSPACE
              </span>
              <ThemeToggle />
            </div>
          </header>

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
