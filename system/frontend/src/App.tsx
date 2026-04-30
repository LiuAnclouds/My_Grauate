import { useMemo, useState } from "react";
import { AuthResponse } from "./services/api";
import { AuthPanel } from "./components/AuthPanel";
import { AuthenticatedAppShell, NavItem } from "./components/layout/AuthenticatedAppShell";

export type AppPage = "monitor" | "access" | "network" | "analysis" | "cases" | "admin";

const navItems: NavItem[] = [
  { key: "monitor", label: "风险工作台", eyebrow: "Overview", description: "查看风险态势与待处理事项", shortLabel: "工作台" },
  { key: "access", label: "业务网络", eyebrow: "Network", description: "接入或切换待分析业务网络", shortLabel: "网络" },
  { key: "network", label: "关系图谱", eyebrow: "Graph", description: "浏览对象关系与交易链路", shortLabel: "图谱" },
  { key: "analysis", label: "智能研判", eyebrow: "Analysis", description: "启动识别任务并跟踪过程", shortLabel: "研判" },
  { key: "cases", label: "风险对象", eyebrow: "Risk", description: "复核异常对象与关联线索", shortLabel: "对象" },
  { key: "admin", label: "系统设置", eyebrow: "System", description: "管理账号、状态与系统配置", shortLabel: "设置" }
];

const operationFlow = [
  { title: "接入业务网络", detail: "上传 CSV 文件并生成可分析的人员关系网络。" },
  { title: "查看关系结构", detail: "在关系网络中定位对象、交易方向和异常关联。" },
  { title: "启动智能研判", detail: "执行特征整理、邻域聚合和模型风险识别。" },
  { title: "复核风险名单", detail: "查看高风险对象画像、分数、原因和关联对象。" }
];

const adminCards = [
  { title: "账号认证", detail: "用户账号、密码哈希和邮箱验证码独立保存，不和业务网络内容混用。" },
  { title: "业务网络", detail: "对象画像、关系边、模型输入和推理结果按业务分析流程组织。" },
  { title: "敏感配置", detail: "邮箱授权码、模型路径等运行配置只保存在本地环境文件。" }
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
