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

import {
  ThemeProvider,
  useTheme,
} from "@/components/theme-provider";

import AppSidebar, {
  type PageId,
} from "@/components/layout/AppSidebar";

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
          aria-label="Change theme"
        >
          <Sun className="size-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute size-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
        </Button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}>
          <Sun className="size-4" />
          Light
          {theme === "light" && (
            <span className="ml-auto">✓</span>
          )}
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => setTheme("dark")}>
          <Moon className="size-4" />
          Dark
          {theme === "dark" && (
            <span className="ml-auto">✓</span>
          )}
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => setTheme("system")}>
          <Monitor className="size-4" />
          System
          {theme === "system" && (
            <span className="ml-auto">✓</span>
          )}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default function App() {
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
    <ThemeProvider defaultTheme="system">
      <TooltipProvider>
        <SidebarProvider>
          <AppSidebar
            activePage={activePage}
            onNavigate={setActivePage}
          />

          <SidebarInset>
            <header className="flex h-14 shrink-0 items-center gap-3 border-b px-4">
              <SidebarTrigger />

              <Separator
                orientation="vertical"
                className="h-5"
              />

              <h1 className="text-sm font-medium">
                {pageTitles[activePage]}
              </h1>

              <div className="ml-auto">
                <ThemeToggle />
              </div>
            </header>

            <main className="min-w-0 flex-1 overflow-auto p-6">
              {renderPage()}
            </main>
          </SidebarInset>
        </SidebarProvider>
      </TooltipProvider>
    </ThemeProvider>
  );
}
