import { useEffect, useMemo, useState } from "react";
import { AuthResponse } from "./services/api";
import { AuthPanel } from "./components/AuthPanel";
import { AuthenticatedAppShell, NavItem } from "./components/layout/AuthenticatedAppShell";
import { AdminView } from "./components/views/AdminView";

export type AppPage = "monitor" | "access" | "network" | "analysis" | "cases";

const navItems: NavItem[] = [
  { key: "monitor", label: "风险工作台", eyebrow: "Overview", description: "掌握风险态势", shortLabel: "工作台" },
  { key: "access", label: "业务网络", eyebrow: "Network", description: "接入与切换网络", shortLabel: "网络" },
  { key: "network", label: "关系图谱", eyebrow: "Graph", description: "查看关系链路", shortLabel: "图谱" },
  { key: "analysis", label: "智能研判", eyebrow: "Analysis", description: "执行风险识别", shortLabel: "研判" },
  { key: "cases", label: "风险对象", eyebrow: "Risk", description: "复核异常对象", shortLabel: "对象" }
];

const operationFlow = [
  { title: "接入业务网络", detail: "上传业务记录，形成网络。" },
  { title: "查看关系结构", detail: "定位对象与交易方向。" },
  { title: "启动智能研判", detail: "识别高风险对象。" },
  { title: "复核风险名单", detail: "核查对象与关联线索。" }
];

const SESSION_STORAGE_KEY = "starhubgraph.auth.session";
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;

type StoredSession = {
  session: AuthResponse;
  expiresAt: number;
};

function fallbackExpiry() {
  return Date.now() + SESSION_TTL_MS;
}

function sessionExpiry(session: AuthResponse) {
  const serverExpiry = session.session_expires_at ? Date.parse(session.session_expires_at) : NaN;
  return Number.isFinite(serverExpiry) ? serverExpiry : fallbackExpiry();
}

function loadStoredSession(): AuthResponse | null {
  try {
    const raw = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;
    const stored = JSON.parse(raw) as Partial<StoredSession>;
    if (!stored.session || typeof stored.expiresAt !== "number" || stored.expiresAt <= Date.now()) {
      window.localStorage.removeItem(SESSION_STORAGE_KEY);
      return null;
    }
    return { ...stored.session, session_expires_at: new Date(stored.expiresAt).toISOString() };
  } catch {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
    return null;
  }
}

function saveStoredSession(session: AuthResponse) {
  const expiresAt = sessionExpiry(session);
  const stored: StoredSession = {
    session: { ...session, session_expires_at: new Date(expiresAt).toISOString() },
    expiresAt
  };
  window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(stored));
  return stored.session;
}

function clearStoredSession() {
  window.localStorage.removeItem(SESSION_STORAGE_KEY);
}

export default function App() {
  const [session, setSession] = useState<AuthResponse | null>(() => loadStoredSession());
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [selectedNetworkName, setSelectedNetworkName] = useState("");
  const [graphRefreshKey, setGraphRefreshKey] = useState(0);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);
  const [activeTimelineNodeId, setActiveTimelineNodeId] = useState<string | null>(null);

  const currentNetwork = useMemo(() => {
    return selectedNetworkName || (selectedDatasetId ? "已接入业务网络" : "尚未接入业务网络");
  }, [selectedDatasetId, selectedNetworkName]);

  function handleBusinessSelect(datasetId: number | null, networkName?: string) {
    setSelectedDatasetId(datasetId);
    setSelectedNetworkName(datasetId ? networkName ?? "已接入业务网络" : "");
    setHighlightedNodeId(null);
    setActiveTimelineNodeId(null);
  }

  function handleAuthed(nextSession: AuthResponse) {
    setSession(saveStoredSession(nextSession));
  }

  function handleLogout() {
    clearStoredSession();
    setSession(null);
  }

  useEffect(() => {
    if (!session) return;
    const expiresAt = sessionExpiry(session);
    const delay = Math.max(0, expiresAt - Date.now());
    const timer = window.setTimeout(() => {
      clearStoredSession();
      setSession(null);
    }, delay);
    return () => window.clearTimeout(timer);
  }, [session]);

  if (!session) {
    return <AuthPanel onAuthed={handleAuthed} />;
  }

  if (session.is_admin) {
    return <AdminView session={session} onLogout={handleLogout} />;
  }

  return (
    <AuthenticatedAppShell
      session={session}
      navItems={navItems}
      currentNetwork={currentNetwork}
      selectedDatasetId={selectedDatasetId}
      graphRefreshKey={graphRefreshKey}
      highlightedNodeId={highlightedNodeId}
      activeTimelineNodeId={activeTimelineNodeId}
      operationFlow={operationFlow}
      onLogout={handleLogout}
      onBusinessSelect={handleBusinessSelect}
      onGraphRefresh={() => setGraphRefreshKey((value) => value + 1)}
      onHighlightNode={setHighlightedNodeId}
      onTimelineNode={setActiveTimelineNodeId}
    />
  );
}
