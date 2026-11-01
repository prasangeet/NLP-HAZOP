import {
  LayoutDashboard,
  Database,
  FileSearch,
  GitBranch,
  Workflow,
  FlaskConical,\n  ChartNoAxesCombined,
} from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";

export type PageId =
  | "overview"
  | "datasets"
  | "report-extraction"
  | "knowledge-graph"
  | "relationship-recovery"
  | "experiments"\n  | "results";

interface AppSidebarProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
}

const navigation = [
  { label: "Overview", page: "overview", icon: LayoutDashboard },
  { label: "Datasets", page: "datasets", icon: Database },
  { label: "Report Extraction", page: "report-extraction", icon: FileSearch },
  { label: "Knowledge Graph", page: "knowledge-graph", icon: GitBranch },
  {
    label: "Relationship Recovery",
    page: "relationship-recovery",
    icon: Workflow,
  },
  { label: "Experiments", page: "experiments", icon: FlaskConical },\n  { label: "Research results", page: "results", icon: ChartNoAxesCombined },
] as const;

export default function AppSidebar({
  activePage,
  onNavigate,
}: AppSidebarProps) {
  return (
    <Sidebar collapsible="icon" variant="sidebar">
      <SidebarHeader className="h-16 justify-center overflow-hidden group-data-[collapsible=icon]:items-center group-data-[collapsible=icon]:px-0">
        <div className="flex w-full min-w-0 items-center gap-3 px-2 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-0">
          <div className="flex size-9 shrink-0 items-center justify-center border border-sidebar-border bg-sidebar text-sidebar-foreground">
            <GitBranch className="size-4 shrink-0" />
          </div>

          <div className="min-w-0 group-data-[collapsible=icon]:hidden">
            <h1 className="truncate text-xs font-semibold tracking-tight">
              NLP-HAZOP
            </h1>
            <p className="truncate text-[10px] text-sidebar-foreground/65">
              Causal reasoning
            </p>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Workspace</SidebarGroupLabel>

          <SidebarGroupContent>
            <SidebarMenu>
              {navigation.map((item) => {
                const Icon = item.icon;

                return (
                  <SidebarMenuItem key={item.page}>
                    <SidebarMenuButton
                      isActive={activePage === item.page}
                      tooltip={item.label}
                      onClick={() => onNavigate(item.page)}
                    >
                      <Icon />
                      <span>{item.label}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="overflow-hidden">
        <div className="border-t border-sidebar-border px-2 py-3 text-[10px] text-sidebar-foreground/60 group-data-[collapsible=icon]:hidden">
          NLP-HAZOP · Research build
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
