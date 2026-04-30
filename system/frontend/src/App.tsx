import { useMemo, useState } from "react";
import { AuthResponse } from "./services/api";
import { AuthPanel } from "./components/AuthPanel";
import { AuthenticatedAppShell, NavItem } from "./components/layout/AuthenticatedAppShell";

export type AppPage = "monitor" | "access" | "network" | "analysis" | "cases" | "admin";

const navItems: NavItem[] = [
  { key: "monitor", label: "风险工作台", eyebrow: "Overview", description: "掌握风险态势", shortLabel: "工作台" },
  { key: "access", label: "业务网络", eyebrow: "Network", description: "接入与切换网络", shortLabel: "网络" },
  { key: "network", label: "关系图谱", eyebrow: "Graph", description: "查看关系链路", shortLabel: "图谱" },
  { key: "analysis", label: "智能研判", eyebrow: "Analysis", description: "执行风险识别", shortLabel: "研判" },
  { key: "cases", label: "风险对象", eyebrow: "Risk", description: "复核异常对象", shortLabel: "对象" },
  { key: "admin", label: "系统设置", eyebrow: "System", description: "账号与配置", shortLabel: "设置" }
];

const operationFlow = [
  { title: "接入业务网络", detail: "上传业务记录，形成网络。" },
  { title: "查看关系结构", detail: "定位对象与交易方向。" },
  { title: "启动智能研判", detail: "识别高风险对象。" },
  { title: "复核风险名单", detail: "核查对象与关联线索。" }
];

const adminCards = [
  { title: "账号认证", detail: "账号与验证码独立管理。" },
  { title: "业务网络", detail: "网络、对象、关系分区保存。" },
  { title: "敏感配置", detail: "运行密钥仅存本地。" }
];

export default function App() {
  const [session, setSession] = useState<AuthResponse | null>(null);
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

  if (!session) {
    return <AuthPanel onAuthed={setSession} />;
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
      adminCards={adminCards}
      onLogout={() => setSession(null)}
      onBusinessSelect={handleBusinessSelect}
      onGraphRefresh={() => setGraphRefreshKey((value) => value + 1)}
      onHighlightNode={setHighlightedNodeId}
      onTimelineNode={setActiveTimelineNodeId}
    />
  );
}
