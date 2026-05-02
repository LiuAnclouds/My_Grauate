import { useEffect, useMemo, useState } from "react";
import { AppPage } from "../../App";
import { AuthResponse } from "../../services/api";
import { DataUpload } from "../DataUpload";
import { GraphWorkspace } from "../GraphWorkspace";
import { InferenceResults } from "../InferenceResults";
import { PipelinePanel } from "../PipelinePanel";
import type { AnalysisFrame } from "../PipelinePanel";
import { PageSurface } from "./PageSurface";
import { SidebarNav } from "./SidebarNav";
import { TopBrandHeader } from "./TopBrandHeader";
import { MonitorView } from "../views/MonitorView";

export type NavItem = {
  key: AppPage;
  label: string;
  eyebrow: string;
  description: string;
  shortLabel: string;
};

type OperationStep = { title: string; detail: string };

type Props = {
  session: AuthResponse;
  navItems: NavItem[];
  currentNetwork: string;
  selectedDatasetId: number | null;
  graphRefreshKey: number;
  highlightedNodeId: string | null;
  activeTimelineNodeId: string | null;
  operationFlow: OperationStep[];
  onLogout: () => void;
  onBusinessSelect: (datasetId: number | null, networkName?: string) => void;
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
  operationFlow,
  onLogout,
  onBusinessSelect,
  onGraphRefresh,
  onHighlightNode,
  onTimelineNode
}: Props) {
  const [activePage, setActivePage] = useState<AppPage>("monitor");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [analysisFrame, setAnalysisFrame] = useState<AnalysisFrame | null>(null);

  const activeNav = useMemo(() => navItems.find((item) => item.key === activePage) ?? navItems[0], [activePage, navItems]);
  const hasNetwork = Boolean(selectedDatasetId);

  function openPage(nextPage: AppPage) {
    setActivePage(nextPage);
    setMobileNavOpen(false);
    if (nextPage !== "analysis") setAnalysisFrame(null);
  }

  useEffect(() => {
    setAnalysisFrame(null);
  }, [selectedDatasetId]);

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

          <div className="app-page-stack">
            {activePage === "monitor" ? (
              <MonitorView ownerId={session.user_id} currentNetwork={currentNetwork} selectedDatasetId={selectedDatasetId} hasNetwork={hasNetwork} operationFlow={operationFlow} onOpenPage={openPage} />
            ) : null}

            {activePage === "access" ? (
              <PageSurface>
                <DataUpload session={session} selectedDatasetId={selectedDatasetId} onSelect={onBusinessSelect} onOpenPage={openPage} />
              </PageSurface>
            ) : null}

            {activePage === "network" ? (
              <PageSurface>
                <GraphWorkspace
                  datasetId={selectedDatasetId}
                  ownerId={session.user_id}
                  refreshKey={graphRefreshKey}
                  highlightedNodeId={highlightedNodeId}
                  timelineNodeId={activeTimelineNodeId}
                />
              </PageSurface>
            ) : null}

            {activePage === "analysis" ? (
              <PageSurface>
                <div className="analysis-command-grid">
                  <GraphWorkspace
                    datasetId={selectedDatasetId}
                    ownerId={session.user_id}
                    refreshKey={graphRefreshKey}
                    highlightedNodeId={highlightedNodeId}
                    timelineNodeId={activeTimelineNodeId}
                    analysisFrame={analysisFrame}
                    compact
                  />
                  <PipelinePanel
                    datasetId={selectedDatasetId}
                    onFocusNode={onTimelineNode}
                    onFrameChange={setAnalysisFrame}
                    onInferenceComplete={onGraphRefresh}
                  />
                </div>
              </PageSurface>
            ) : null}

            {activePage === "cases" ? (
              <PageSurface>
                <InferenceResults datasetId={selectedDatasetId} refreshKey={graphRefreshKey} onNodeFocus={onHighlightNode} />
              </PageSurface>
            ) : null}

          </div>
        </section>
      </div>
    </main>
  );
}
