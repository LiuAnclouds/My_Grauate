import { useMemo, useState } from "react";
import { AppPage } from "../../App";
import { AuthResponse } from "../../services/api";
import { DataUpload } from "../DataUpload";
import { GraphWorkspace } from "../GraphWorkspace";
import { InferenceResults } from "../InferenceResults";
import { PipelinePanel } from "../PipelinePanel";
import { PageHeader } from "./PageHeader";
import { PageSurface } from "./PageSurface";
import { SidebarNav } from "./SidebarNav";
import { TopBrandHeader } from "./TopBrandHeader";
import { AdminView } from "../views/AdminView";
import { MonitorView } from "../views/MonitorView";

export type NavItem = {
  key: AppPage;
  label: string;
  eyebrow: string;
  description: string;
  shortLabel: string;
};

type MonitorCard = { label: string; value: string; detail: string };
type OperationStep = { title: string; detail: string };
type AdminCard = { title: string; detail: string };

type Props = {
  session: AuthResponse;
  navItems: NavItem[];
  currentNetwork: string;
  selectedDatasetId: number | null;
  graphRefreshKey: number;
  highlightedNodeId: string | null;
  activeTimelineNodeId: string | null;
  monitorCards: MonitorCard[];
  operationFlow: OperationStep[];
  adminCards: AdminCard[];
  onLogout: () => void;
  onBusinessSelect: (datasetId: number, networkName?: string) => void;
  onGraphRefresh: () => void;
  onHighlightNode: (nodeId: string | null) => void;
  onTimelineNode: (nodeId: string | null) => void;
};

export function AuthenticatedAppShell({
  session,
  navItems,
  currentNetwork,
  selectedDatasetId,
  graphRefreshKey,
  highlightedNodeId,
  activeTimelineNodeId,
  monitorCards,
  operationFlow,
  adminCards,
  onLogout,
  onBusinessSelect,
  onGraphRefresh,
  onHighlightNode,
  onTimelineNode
}: Props) {
  const [activePage, setActivePage] = useState<AppPage>("monitor");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const activeNav = useMemo(() => navItems.find((item) => item.key === activePage) ?? navItems[0], [activePage, navItems]);
  const hasNetwork = Boolean(selectedDatasetId);

  function openPage(nextPage: AppPage) {
    setActivePage(nextPage);
    setMobileNavOpen(false);
  }

  return (
    <main className="app-shell enterprise-app-shell nextgen-shell">
      <div className={sidebarCollapsed ? "app-shell-grid is-collapsed" : "app-shell-grid"}>
        <SidebarNav
          items={navItems}
          activePage={activePage}
          collapsed={sidebarCollapsed}
          mobileOpen={mobileNavOpen}
          onSelect={openPage}
          onToggleCollapse={() => setSidebarCollapsed((value) => !value)}
          onCloseMobile={() => setMobileNavOpen(false)}
        />

        <section className="app-workspace">
          <TopBrandHeader
            session={session}
            currentNetwork={currentNetwork}
            currentPageLabel={activeNav.label}
            currentPageDescription={activeNav.description}
            onOpenMobileNav={() => setMobileNavOpen(true)}
            onLogout={onLogout}
          />

          <PageHeader
            eyebrow={activeNav.eyebrow}
            title={activeNav.label}
            description={activeNav.description}
            status={hasNetwork ? "业务网络已接入" : "等待业务网络接入"}
            meta={[
              { label: "工作模式", value: activePage === "analysis" ? "研判工作台" : "系统工作区" },
              { label: "用户角色", value: session.is_admin ? "管理员" : "分析员" }
            ]}
          />

          <div className="app-page-stack">
            {activePage === "monitor" ? (
              <MonitorView currentNetwork={currentNetwork} hasNetwork={hasNetwork} monitorCards={monitorCards} operationFlow={operationFlow} onOpenPage={openPage} />
            ) : null}

            {activePage === "access" ? (
              <PageSurface>
                <DataUpload selectedDatasetId={selectedDatasetId} onSelect={onBusinessSelect} onOpenPage={openPage} />
              </PageSurface>
            ) : null}

            {activePage === "network" ? (
              <PageSurface>
                <GraphWorkspace
                  datasetId={selectedDatasetId}
                  refreshKey={graphRefreshKey}
                  highlightedNodeId={highlightedNodeId}
                  timelineNodeId={activeTimelineNodeId}
                />
              </PageSurface>
            ) : null}

            {activePage === "analysis" ? (
              <PageSurface>
                <div className="analysis-command-grid">
                  <PipelinePanel datasetId={selectedDatasetId} onFocusNode={onTimelineNode} onInferenceComplete={onGraphRefresh} />
                  <GraphWorkspace
                    datasetId={selectedDatasetId}
                    refreshKey={graphRefreshKey}
                    highlightedNodeId={highlightedNodeId}
                    timelineNodeId={activeTimelineNodeId}
                    compact
                  />
                </div>
              </PageSurface>
            ) : null}

            {activePage === "cases" ? (
              <PageSurface>
                <InferenceResults datasetId={selectedDatasetId} refreshKey={graphRefreshKey} onNodeFocus={onHighlightNode} />
              </PageSurface>
            ) : null}

            {activePage === "admin" ? (
              <PageSurface>
                <AdminView cards={adminCards} />
              </PageSurface>
            ) : null}
          </div>
        </section>
      </div>
    </main>
  );
}
