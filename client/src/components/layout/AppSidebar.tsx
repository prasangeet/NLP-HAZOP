import {
  LayoutDashboard,
  Database,
  FileSearch,
  GitBranch,
  Workflow,
  FlaskConical,
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
  | "experiments";

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
  { label: "Experiments", page: "experiments", icon: FlaskConical },
] as const;

export default function AppSidebar({
  activePage,
  onNavigate,
}: AppSidebarProps) {
  return (
    <Sidebar collapsible="icon" variant="sidebar">
      <SidebarHeader className="h-16 justify-center overflow-hidden group-data-[collapsible=icon]:items-center">
        <div className="flex min-w-0 items-center gap-3 px-2 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-0">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
            <GitBranch className="size-5 shrink-0" />
          </div>

          <div className="min-w-0 group-data-[collapsible=icon]:hidden">
            <h1 className="truncate text-sm font-semibold tracking-tight">
              NLP-HAZOP
            </h1>
            <p className="truncate text-xs text-sidebar-foreground/70">
              Causal reasoning
            </p>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent className="overflow-x-hidden">
        <SidebarGroup>
          <SidebarGroupLabel>Workspace</SidebarGroupLabel>

          <SidebarGroupContent>
            <SidebarMenu>
              {navigation.map((item) => {
                const Icon = item.icon;
                const isActive = activePage === item.page;

                return (
                  <SidebarMenuItem key={item.page}>
                    <SidebarMenuButton
                      isActive={isActive}
                      tooltip={item.label}
                      onClick={() => onNavigate(item.page)}
                    >
                      <Icon className="shrink-0" />
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
        <div className="px-2 py-2 text-xs text-sidebar-foreground/70 group-data-[collapsible=icon]:hidden">
          Research build
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
